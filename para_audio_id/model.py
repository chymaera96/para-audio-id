from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import nn

from .codes import BOS_TOKEN, CODE_LENGTH, VOCAB_SIZE, teacher_forcing_inputs, tokens_to_code


class MuQEncoder(nn.Module):
    def __init__(self, model_name: str, encoder_dim: int = 1024):
        super().__init__()
        try:
            from muq import MuQ
        except ImportError as exc:
            raise ImportError("Install the project with its `muq` dependency to use MuQEncoder") from exc
        self.model = MuQ.from_pretrained(model_name)
        self.encoder_dim = encoder_dim
        self._upper_blocks: list[nn.Module] = []
        self._fully_frozen = False

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 2:
            raise ValueError(f"Expected MuQ input [batch, time], got {tuple(audio.shape)}")
        context = torch.no_grad() if self._fully_frozen else nullcontext()
        with context:
            output = self.model(audio, output_hidden_states=True)
        hidden = output.last_hidden_state
        if hidden.ndim != 3 or hidden.shape[0] != audio.shape[0]:
            raise ValueError(
                "Expected MuQ output [batch, sequence, hidden], "
                f"got {tuple(hidden.shape)}"
            )
        if hidden.shape[-1] != self.encoder_dim:
            raise ValueError(
                f"Expected MuQ hidden size {self.encoder_dim}, got {hidden.shape[-1]}"
            )
        return hidden

    def transformer_blocks(self) -> list[nn.Module]:
        candidates: list[nn.ModuleList] = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ModuleList) and any(word in name.lower() for word in ("layer", "block")):
                candidates.append(module)
        if not candidates:
            raise RuntimeError("Could not locate MuQ transformer blocks")
        return list(max(candidates, key=len))

    def freeze_all(self) -> None:
        self.model.requires_grad_(False)
        self._upper_blocks = []
        self._fully_frozen = True
        self.model.eval()

    def unfreeze_upper_fraction(self, fraction: float = 0.25) -> list[nn.Module]:
        if not 0 < fraction <= 1:
            raise ValueError(f"Unfreeze fraction must be in (0, 1], got {fraction}")
        blocks = self.transformer_blocks()
        count = max(1, round(len(blocks) * fraction))
        self._upper_blocks = blocks[-count:]
        for block in self._upper_blocks:
            block.requires_grad_(True)
        self._fully_frozen = False
        self.model.eval()
        for block in self._upper_blocks:
            block.train(self.training)
        return self._upper_blocks

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen MuQ components must never acquire train-mode stochasticity.
        self.model.eval()
        if not self._fully_frozen:
            for block in self._upper_blocks:
                block.train(mode)
        return self

    @property
    def fully_frozen(self) -> bool:
        return self._fully_frozen


class DigitDecoder(nn.Module):
    def __init__(
        self,
        *,
        encoder_dim: int = 1024,
        model_dim: int = 512,
        layers: int = 6,
        heads: int = 8,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(encoder_dim, model_dim), nn.LayerNorm(model_dim))
        self.tokens = nn.Embedding(VOCAB_SIZE, model_dim)
        self.positions = nn.Embedding(CODE_LENGTH, model_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers, norm=nn.LayerNorm(model_dim))
        self.output = nn.Linear(model_dim, VOCAB_SIZE)

    def forward(self, hidden: torch.Tensor, input_tokens: torch.Tensor) -> torch.Tensor:
        memory = self.projection(hidden)
        positions = torch.arange(input_tokens.shape[1], device=input_tokens.device)
        target = self.tokens(input_tokens) + self.positions(positions).unsqueeze(0)
        causal = nn.Transformer.generate_square_subsequent_mask(
            input_tokens.shape[1], device=input_tokens.device
        )
        decoded = self.decoder(target, memory, tgt_mask=causal, tgt_is_causal=True)
        return self.output(decoded)


@dataclass(frozen=True)
class BeamResult:
    code: str
    log_probability: float


class ParametricAudioIdentifier(nn.Module):
    def __init__(self, cfg: dict, encoder: nn.Module | None = None):
        super().__init__()
        model_cfg = cfg["model"]
        self.muq = encoder or MuQEncoder(
            model_cfg["muq_name"], encoder_dim=int(model_cfg["encoder_dim"])
        )
        decoder_cfg = model_cfg["decoder"]
        self.digit_decoder = DigitDecoder(
            encoder_dim=int(model_cfg["encoder_dim"]),
            model_dim=int(decoder_cfg["model_dim"]),
            layers=int(decoder_cfg["layers"]),
            heads=int(decoder_cfg["heads"]),
            feedforward_dim=int(decoder_cfg["feedforward_dim"]),
            dropout=float(decoder_cfg["dropout"]),
        )

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        return self.muq(audio)

    def forward(self, audio: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.digit_decoder(self.encode(audio), teacher_forcing_inputs(targets))

    @torch.no_grad()
    def greedy_decode(self, audio: torch.Tensor) -> list[str]:
        hidden = self.encode(audio)
        tokens = torch.full(
            (audio.shape[0], 1), BOS_TOKEN, dtype=torch.long, device=audio.device
        )
        generated = []
        for _ in range(CODE_LENGTH):
            logits = self.digit_decoder(hidden, tokens)[:, -1]
            logits[:, BOS_TOKEN] = -torch.inf
            digit = logits.argmax(dim=-1)
            generated.append(digit)
            tokens = torch.cat((tokens, digit[:, None]), dim=1)
        matrix = torch.stack(generated, dim=1)
        return [tokens_to_code(row) for row in matrix]

    @torch.no_grad()
    def beam_decode(self, audio: torch.Tensor, width: int = 10) -> list[list[BeamResult]]:
        hidden_batch = self.encode(audio)
        all_results: list[list[BeamResult]] = []
        for hidden in hidden_batch:
            beams: list[tuple[list[int], float]] = [([BOS_TOKEN], 0.0)]
            for _ in range(CODE_LENGTH):
                expanded: list[tuple[list[int], float]] = []
                for prefix, score in beams:
                    inputs = torch.tensor(prefix, device=hidden.device).unsqueeze(0)
                    logits = self.digit_decoder(hidden.unsqueeze(0), inputs)[:, -1]
                    logits[:, BOS_TOKEN] = -torch.inf
                    log_probs = logits.log_softmax(dim=-1)
                    values, indices = log_probs.topk(min(width, 10))
                    expanded.extend(
                        (prefix + [int(token)], score + float(value))
                        for value, token in zip(values[0], indices[0], strict=True)
                    )
                beams = sorted(expanded, key=lambda item: item[1], reverse=True)[:width]
            unique: dict[str, float] = {}
            for sequence, score in beams:
                code = tokens_to_code(sequence[1:])
                unique[code] = max(score, unique.get(code, -torch.inf))
            all_results.append(
                [BeamResult(code, score) for code, score in sorted(unique.items(), key=lambda x: -x[1])]
            )
        return all_results
