"""Fixed FSQ replay and source-decision preservation for continual arms D/E."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from ..catalogue import CatalogueRecord
from .checkpoint import load_audio_lm
from .continual import (AdditionCollator, asset_loaders, digest, file_digest,
                        full_schedule, write_json)
from .dataset import collate_causal_documents
from .evaluation import _checkpoint_tokenizer
from .generation import batched_beam_generate, prompts_from_audio_tokens
from .noise import stable_uint64
from .random_crops import OnlineTrackDataset

REPLAY_PROTOCOL = "fsq_fixed_replay_v1"


def identifier_positions(documents, vocabulary):
    """Five teacher-forced digit positions followed by EOS (causally shifted)."""
    targets = documents["input_ids"][:, 1:]
    mask = documents["id_target_mask"] | (targets == vocabulary.eos_token_id)
    if not bool((mask.sum(1) == 6).all()):
        raise ValueError("Replay preservation expects five ID digits and EOS")
    return mask


def grouped_log_probs(logits, vocabulary):
    """Keep 10 digits, EOS, and total remaining-vocabulary mass.

    This preserves the full-softmax normalizer, unlike a digit-only softmax.
    It is a grouped KL proxy, not full-vocabulary or free-running sequence KL.
    """
    values = logits.float()
    ids = list(vocabulary.digit_token_ids) + [vocabulary.eos_token_id]
    rest = torch.ones(values.shape[-1], device=values.device, dtype=torch.bool)
    rest[ids] = False
    groups = torch.cat([values[..., ids], values[..., rest].logsumexp(-1, keepdim=True)], -1)
    return groups.log_softmax(-1)


def preservation_loss(logits, documents, vocabulary, teacher, eligible, id_weight=16.0):
    selected = logits[:, :-1][identifier_positions(documents, vocabulary)]
    student = grouped_log_probs(selected, vocabulary).reshape(len(logits), 6, 12)
    target = teacher.float().log_softmax(-1)
    divergence = (target.exp() * (target - student)).sum(-1)
    weights = divergence.new_tensor([id_weight] * 5 + [1.0])
    per_document = (divergence * weights).sum(-1) / weights.sum()
    # Mean over ALL documents: new documents and teacher-incorrect old ones
    # contribute zero, rather than changing the coefficient as eligibility varies.
    return (per_document * eligible.to(per_document.dtype)).mean()


class ReplayCache:
    def __init__(self, directory, pilot):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        if (self.manifest["protocol"] != REPLAY_PROTOCOL
                or self.manifest["contract"]["pilot_sha256"] != digest(pilot)):
            raise ValueError("Replay cache belongs to a different pilot")
        parts = []
        for item in self.manifest["shards"]:
            path = directory / item["name"]
            if file_digest(path) != item["sha256"]:
                raise ValueError(f"Changed replay shard: {path}")
            parts.append(torch.load(path, map_location="cpu", weights_only=False))
        self.tokens = torch.cat([p["tokens"] for p in parts])
        self.teacher = torch.cat([p["teacher"] for p in parts])
        self.eligible = torch.cat([p["eligible"] for p in parts])
        self.rows = [r for p in parts for r in p["rows"]]
        if (len(self.rows) % 2 or len(self.tokens) != len(self.rows)
                or self.teacher.shape != (len(self.rows), 6, 12)
                or self.eligible.shape != (len(self.rows),)):
            raise ValueError("Incomplete replay documents or teacher targets")
        self.pairs = defaultdict(list)
        for index in range(0, len(self.rows), 2):
            left, right = self.rows[index:index + 2]
            if left["track_id"] != right["track_id"]:
                raise ValueError("Replay pair identity mismatch")
            self.pairs[left["track_id"]].append(index)
        self.sha256 = digest(self.manifest)


class CachedAdditionCollator:
    """Decode newcomers only; old observations come from the shared token cache."""
    def __init__(self, new, replay, cache, **kwargs):
        self.fresh = AdditionCollator(new, [], **kwargs)
        self.new_ids = {r.track_id for r in new}
        self.cache = cache
        self.seed = kwargs["seed"]

    def __call__(self, examples):
        fresh = [e for e in examples if e["record"].track_id in self.new_ids]
        old = [e for e in examples if e["record"].track_id not in self.new_ids]
        result = self.fresh(fresh)
        indices = []
        for e in old:
            choices = self.cache.pairs[e["record"].track_id]
            key = stable_uint64(self.seed, e["optimizer_step"], e["batch_idx"],
                                e["pair_slot"], "cached-replay-view")
            first = choices[key % len(choices)]
            indices.extend([first, first + 1])
        result["cached_tokens"] = self.cache.tokens[indices].long()
        result["teacher"] = self.cache.teacher[indices]
        result["eligible"] = self.cache.eligible[indices]
        result["metadata"].extend(dict(self.cache.rows[i]) for i in indices)
        result["categories"] = dict(Counter(result["categories"]) + Counter(
            self.cache.rows[i]["category"] for i in indices[1::2]))
        return result


def prepare_replay(pilot_directory, output, *, workers=8, pairs_per_track=4):
    """Resumable shards; final manifest seals cache and teacher targets together."""
    from tqdm import tqdm

    started = time.perf_counter()
    pilot = json.loads((Path(pilot_directory) / "manifest.json").read_text())
    if file_digest(pilot["source_checkpoint"]) != pilot["source_sha256"]:
        raise ValueError("Source checkpoint changed")
    output = Path(output)
    contract = dict(protocol=REPLAY_PROTOCOL, pilot_sha256=digest(pilot),
                    pairs_per_track=pairs_per_track, pairs_per_batch=32,
                    seed=pilot["seed"] + 991, teacher="original_tc21_fp32_eval",
                    gate="width10_eos_top1_correct", targets="digits_eos_other_grouped_kl_v1")
    if (output / "manifest.json").exists():
        existing = ReplayCache(output, pilot)
        if existing.manifest["contract"] != contract:
            raise ValueError("Existing replay preparation contract differs")
        print("Using verified complete replay cache:", output, flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    if (output / "preparation.json").exists():
        if json.loads((output / "preparation.json").read_text()) != contract:
            raise ValueError("Partial replay preparation contract differs")
    else:
        write_json(output / "preparation.json", contract)
    model, vocabulary, cfg, source = load_audio_lm(pilot["source_checkpoint"], "cpu")
    model.cuda().eval()
    tokenizer = _checkpoint_tokenizer(source, "cuda")
    if source["tokenizer_fingerprint"] != pilot["tokenizer_fingerprint"]:
        raise ValueError("Source tokenizer differs from pilot")
    records = [CatalogueRecord(**r) for r in pilot["old"]]
    noise, room = asset_loaders(cfg, tokenizer.sample_rate)
    if noise.manifest() != pilot["noise_manifest"] or room.manifest() != pilot["rir_manifest"]:
        raise ValueError("Training degradation assets changed")
    batches = []
    shard_numbers = []
    shard = 0
    for view in range(pairs_per_track):
        for start in range(0, len(records), 32):
            if not (output / f"shard-{shard:05d}.pt").exists():
                batches.append([(i, view, shard, slot)
                                for slot, i in enumerate(range(start, min(start + 32, len(records))))])
                shard_numbers.append(shard)
            shard += 1
    collator = AdditionCollator(records, [], audio_root=cfg["data"]["audio_root"],
        noise_assets=noise, rir_assets=room, schedule=full_schedule(source),
        sample_rate=tokenizer.sample_rate, crop_duration=2.0, past_context_duration=2.0,
        seed=contract["seed"], reserved_starts=pilot["reserved_starts"])
    loader = DataLoader(OnlineTrackDataset(records), batch_sampler=batches,
        collate_fn=collator, num_workers=workers,
        multiprocessing_context="spawn" if workers else None,
        persistent_workers=bool(workers), prefetch_factor=2 if workers else None)
    for number, batch in tqdm(zip(shard_numbers, loader, strict=True), total=len(batches),
                              desc="Preparing fixed old replay"):
        with torch.inference_mode():
            tokens = tokenizer.tokenize(batch["waveforms"].cuda()).long()
            documents = collate_causal_documents([
                dict(audio_tokens=t, code=r["code"], track_id=r["track_id"],
                     document_index=-1, view_type="noisy" if r["is_noisy"] else "random_clean")
                for t, r in zip(tokens, batch["metadata"], strict=True)], vocabulary, 512)
            documents = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                         for k, v in documents.items()}
            logits = model(documents["input_ids"], documents["attention_mask"])
            teacher = grouped_log_probs(logits[:, :-1][identifier_positions(documents, vocabulary)],
                                        vocabulary).reshape(len(tokens), 6, 12).cpu().half()
            del logits, documents
            predictions = []
            for offset in range(0, len(tokens), 16):
                beams = batched_beam_generate(model,
                    prompts_from_audio_tokens(tokens[offset:offset + 16], vocabulary),
                    vocabulary, width=10, score_eos=True)
                predictions.extend(b[0].code for b in beams)
        rows = batch["metadata"]
        for row, prediction in zip(rows, predictions, strict=True):
            row["teacher_prediction"] = prediction
        eligible = torch.tensor([p == r["code"] for p, r in zip(predictions, rows, strict=True)])
        payload = dict(tokens=tokens.cpu().to(torch.int16), teacher=teacher,
                       eligible=eligible, rows=rows, failures=batch["failures"],
                       replacements=batch["replacements"], contract_sha256=digest(contract))
        path = output / f"shard-{number:05d}.pt"
        torch.save(payload, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)
    coverage, gates, categories, shards, failures = Counter(), Counter(), Counter(), [], []
    for number in range(shard):
        path = output / f"shard-{number:05d}.pt"
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data["contract_sha256"] != digest(contract):
            raise ValueError("Replay shard preparation contract differs")
        coverage.update(r["track_id"] for r in data["rows"][::2])
        categories.update(r["category"] for r in data["rows"][1::2])
        for r, ok in zip(data["rows"], data["eligible"].tolist(), strict=True):
            label = "degraded" if r["is_noisy"] else "clean"
            gates[label + "_total"] += 1
            gates[label + "_correct"] += int(ok)
        failures.extend(data["failures"])
        shards.append(dict(name=path.name, sha256=file_digest(path)))
    uncovered = [r.track_id for r in records if not coverage[r.track_id]]
    if len(uncovered) > .01 * len(records):
        raise RuntimeError(f"Replay coverage unexpectedly low: {len(uncovered)} missing identities")
    manifest = dict(protocol=REPLAY_PROTOCOL, contract=contract, shards=shards,
        covered_tracks=len(coverage), requested_tracks=len(records), uncovered_ids=uncovered,
        pairs=sum(coverage.values()), pairs_per_identity=dict(coverage),
        categories=dict(categories), teacher_gate_counts=dict(gates), failures=failures,
        preparation_seconds=time.perf_counter() - started,
        total_bytes=sum((output / x["name"]).stat().st_size for x in shards))
    write_json(output / "manifest.json", manifest)
    print("REPLAY READY", json.dumps({k: v for k, v in manifest.items()
          if k not in ("shards", "pairs_per_identity", "failures")}), flush=True)
