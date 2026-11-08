"""Independent write-locality, constrained-update and capacity experiments."""
from __future__ import annotations

import itertools
import math

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize

from .continual_replay import identifier_positions
from .dataset import collate_causal_documents
from .noise import stable_uint64


class LowRankWeight(nn.Module):
    """GPT-2 Conv1D has [input, output] weights, unlike nn.Linear."""
    def __init__(self, weight, rank=32):
        super().__init__()
        self.a = nn.Parameter(weight.new_empty(weight.shape[0], rank))
        self.b = nn.Parameter(weight.new_zeros(rank, weight.shape[1]))
        nn.init.kaiming_uniform_(self.a.T, a=math.sqrt(5))

    def forward(self, original):
        return original + self.a @ self.b  # alpha=rank, hence scale=1


def configure_write_parameters(model, arm, *, rank=32, saved_adapters=None):
    if arm == "F":
        model.requires_grad_(False)
        for block in model.network.transformer.h[-4:]:
            block.mlp.requires_grad_(True)
    elif arm == "I":
        model.requires_grad_(False)
        for block in model.network.transformer.h:
            for module in (block.attn.c_attn, block.attn.c_proj, block.mlp.c_fc, block.mlp.c_proj):
                parametrize.register_parametrization(module, "weight", LowRankWeight(module.weight, rank))
        if saved_adapters is not None:
            parameters = dict(model.named_parameters())
            expected = {name for name, p in parameters.items() if p.requires_grad}
            if set(saved_adapters) != expected:
                raise ValueError("LoRA checkpoint adapter names differ")
            with torch.no_grad():
                for name, value in saved_adapters.items():
                    parameters[name].copy_(value)
    return [p for p in model.parameters() if p.requires_grad]


def exported_state(model):
    """Merged weights load through ordinary load_audio_lm; adapters resume separately."""
    state = {k: v for k, v in model.state_dict().items() if ".parametrizations." not in k}
    adapters = {}
    for name, module in model.named_modules():
        if parametrize.is_parametrized(module, "weight"):
            state[name + ".weight"] = module.weight.detach().cpu()
    if any(parametrize.is_parametrized(m, "weight") for m in model.modules()):
        adapters = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    return state, adapters


def projection_multipliers(gram, products, *, tolerance=1e-9):
    """Exact active-set solve for <=3 normalized half-space constraints.

    Project displacement d onto A d <= 0: d' = d - A.T lambda.
    Enumerating eight active sets avoids an external QP dependency.
    """
    gram, products = np.asarray(gram, dtype=np.float64), np.asarray(products, dtype=np.float64)
    n = len(products)
    if n > 3:
        raise ValueError("This pilot has at most three reference conditions")
    best, best_cost = None, float("inf")
    for count in range(n + 1):
        for active in itertools.combinations(range(n), count):
            multipliers = np.zeros(n)
            if active:
                ids = list(active)
                multipliers[ids] = np.linalg.lstsq(gram[np.ix_(ids, ids)], products[ids], rcond=1e-12)[0]
            residual = products - gram @ multipliers
            if np.min(multipliers, initial=0) < -tolerance or np.max(residual, initial=0) > tolerance:
                continue
            cost = float(multipliers @ gram @ multipliers)
            if cost < best_cost:
                best, best_cost = multipliers, cost
    return best


def flat_parameters(parameters):
    return torch.cat([p.detach().reshape(-1) for p in parameters])


@torch.no_grad()
def constrain_adam_displacement(parameters, before, references):
    after = flat_parameters(parameters)
    displacement = after - before
    norm = float(displacement.norm())
    rows = []
    for gradient in references:
        size = gradient.norm()
        if float(size) > 1e-12:
            rows.append(gradient / size)
    if not rows:
        return dict(projection_correction_ratio=0.0, projection_actual_max_violation=0.0,
                    projection_proposed_max_violation=0.0, projection_zero_fallback=0)
    gram = torch.stack([torch.stack([torch.dot(a, b) for b in rows]) for a in rows])
    products = torch.stack([torch.dot(a, displacement) for a in rows])
    multipliers = projection_multipliers(gram.cpu().double().numpy(), products.cpu().double().numpy())
    fallback = int(multipliers is None)
    if multipliers is None:
        corrected = torch.zeros_like(displacement)
    else:
        corrected = displacement.clone()
        for coefficient, row in zip(multipliers, rows, strict=True):
            corrected.add_(row, alpha=-float(coefficient))
    proposed = float(products.max())
    offset = 0
    for p in parameters:
        count = p.numel()
        p.copy_((before[offset:offset + count] + corrected[offset:offset + count]).view_as(p))
        offset += count
    actual = flat_parameters(parameters) - before
    violation = max(float(torch.dot(row, actual)) for row in rows)
    # Parameter rounding can undo a tiny projection; retaining the exact old
    # weights is a feasible fallback. Adam moments still advance explicitly.
    if violation > 1e-7:
        offset = 0
        for p in parameters:
            count = p.numel()
            p.copy_(before[offset:offset + count].view_as(p))
            offset += count
        actual.zero_()
        violation, fallback = 0.0, 1
    return dict(projection_correction_ratio=float((actual - displacement).norm()) / max(norm, 1e-12),
                projection_actual_max_violation=violation,
                projection_proposed_max_violation=proposed,
                projection_zero_fallback=fallback, projection_update_norm=float(actual.norm()))


class ReferenceGradients:
    """Training-cache references only, never the held-out validation queries."""
    def __init__(self, cache, vocabulary, *, count=16, seed=7331):
        self.cache, self.vocabulary, self.count, self.seed = cache, vocabulary, count, seed
        self.pools = {condition: np.asarray([i for i, row in enumerate(cache.rows)
                      if row["category"] == condition]) for condition in ("clean", "noise", "noise_rir")}
        if any(len(pool) == 0 for pool in self.pools.values()):
            raise ValueError("Reference cache lacks a required condition")

    def compute(self, model, parameters, step, dtype):
        model.eval()
        gradients, metrics = [], {}
        try:
            for condition, pool in self.pools.items():
                rng = np.random.default_rng(stable_uint64(self.seed, step, condition, "reference"))
                indices = rng.choice(pool, self.count, replace=len(pool) < self.count)
                rows = [self.cache.rows[int(i)] for i in indices]
                tokens = self.cache.tokens[indices].cuda().long()
                docs = collate_causal_documents([
                    dict(audio_tokens=t, code=r["code"], track_id=r["track_id"], document_index=-1,
                         view_type="noisy" if r["is_noisy"] else "random_clean")
                    for t, r in zip(tokens, rows, strict=True)], self.vocabulary, 512)
                docs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in docs.items()}
                with torch.autocast("cuda", dtype=dtype):
                    logits = model(docs["input_ids"], docs["attention_mask"])
                    mask = identifier_positions(docs, self.vocabulary)
                    chosen = logits[:, :-1][mask].float().reshape(self.count, 6, -1)
                    targets = docs["input_ids"][:, 1:][mask].reshape(self.count, 6)
                    ce = torch.nn.functional.cross_entropy(chosen.transpose(1, 2), targets, reduction="none")
                    weights = ce.new_tensor([16.] * 5 + [1.])
                    loss = (ce * weights).sum(-1).mean() / weights.sum()
                values = torch.autograd.grad(loss, parameters, allow_unused=True)
                flat = torch.cat([(v.detach() if v is not None else torch.zeros_like(p)).reshape(-1)
                                  for v, p in zip(values, parameters, strict=True)])
                if not torch.isfinite(flat).all():
                    raise RuntimeError("Non-finite reference gradient")
                gradients.append(flat)
                metrics[f"reference_{condition}_loss"] = float(loss.detach())
                del logits, docs, chosen, loss, values
        finally:
            model.train()
        return gradients, metrics
