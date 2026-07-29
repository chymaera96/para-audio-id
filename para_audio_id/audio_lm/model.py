from __future__ import annotations

import torch
from torch import nn

from .vocabulary import AudioLMVocabulary


class AudioCausalLM(nn.Module):
    def __init__(self, cfg: dict, vocabulary: AudioLMVocabulary):
        super().__init__()
        try:
            from transformers import GPT2Config, GPT2LMHeadModel
        except ImportError as exc:
            raise ImportError("Install transformers to use the audio causal LM") from exc
        model_cfg = cfg["model"]
        if model_cfg.get("architecture") != "gpt2":
            raise ValueError(f"Unsupported causal architecture {model_cfg.get('architecture')!r}")
        configuration = GPT2Config(
            vocab_size=vocabulary.size,
            n_positions=int(model_cfg["max_position_embeddings"]),
            n_ctx=int(model_cfg["max_position_embeddings"]),
            n_embd=int(model_cfg["hidden_size"]),
            n_layer=int(model_cfg["num_layers"]),
            n_head=int(model_cfg["num_attention_heads"]),
            resid_pdrop=float(model_cfg["resid_pdrop"]),
            embd_pdrop=float(model_cfg["embd_pdrop"]),
            attn_pdrop=float(model_cfg["attn_pdrop"]),
            bos_token_id=vocabulary.bos_token_id,
            eos_token_id=vocabulary.eos_token_id,
            tie_word_embeddings=bool(model_cfg["tie_word_embeddings"]),
            use_cache=False,
        )
        self.network = GPT2LMHeadModel(configuration)
        self.vocabulary = vocabulary

    @property
    def max_position_embeddings(self) -> int:
        return int(self.network.config.n_positions)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_final_hidden_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError(f"Expected [batch, sequence] input IDs, got {input_ids.shape}")
        if input_ids.shape[1] > self.max_position_embeddings:
            raise ValueError("Input exceeds configured positional context")
        if not return_final_hidden_state:
            return self.network(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
        hidden = self.network.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state
        return self.network.lm_head(hidden), hidden

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        *,
        past_key_values=None,
    ):
        """Run autoregressive inference while retaining GPT-2's attention cache."""
        return self.network(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

    @staticmethod
    def reorder_cache(past_key_values, indices: torch.Tensor):
        """Select/repeat cached beams across supported Transformers cache formats."""
        if hasattr(past_key_values, "batch_select_indices"):
            reordered = past_key_values.batch_select_indices(indices)
            return past_key_values if reordered is None else reordered
        if hasattr(past_key_values, "reorder_cache"):
            reordered = past_key_values.reorder_cache(indices)
            return past_key_values if reordered is None else reordered
        return tuple(
            tuple(value.index_select(0, indices) for value in layer)
            for layer in past_key_values
        )
