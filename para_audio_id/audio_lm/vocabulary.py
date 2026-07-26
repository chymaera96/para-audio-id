from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class AudioLMVocabulary:
    num_codebooks: int = 2
    codebook_size: int = 1024

    @property
    def audio_size(self) -> int:
        return self.num_codebooks * self.codebook_size

    @property
    def bos_token_id(self) -> int:
        return self.audio_size

    @property
    def id_token_id(self) -> int:
        return self.audio_size + 1

    @property
    def digit_offset(self) -> int:
        return self.audio_size + 2

    @property
    def eos_token_id(self) -> int:
        return self.audio_size + 12

    @property
    def size(self) -> int:
        return self.audio_size + 13

    @property
    def digit_token_ids(self) -> range:
        return range(self.digit_offset, self.digit_offset + 10)

    def audio_token_id(self, codebook: int, raw_code: int) -> int:
        if not 0 <= codebook < self.num_codebooks:
            raise ValueError(f"Invalid codebook {codebook}")
        if not 0 <= raw_code < self.codebook_size:
            raise ValueError(f"Invalid raw code {raw_code}")
        return codebook * self.codebook_size + raw_code

    def encode_code(self, code: str) -> torch.Tensor:
        if len(code) != 5 or not code.isdecimal():
            raise ValueError(f"Expected five decimal digits, got {code!r}")
        return torch.tensor(
            [self.digit_offset + int(character) for character in code],
            dtype=torch.long,
        )

    def decode_code(self, tokens: torch.Tensor | list[int]) -> str:
        values = tokens.tolist() if isinstance(tokens, torch.Tensor) else list(tokens)
        if len(values) != 5 or any(value not in self.digit_token_ids for value in values):
            raise ValueError(f"Expected five digit tokens, got {values}")
        return "".join(str(value - self.digit_offset) for value in values)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> AudioLMVocabulary:
        return cls(**value)

    def validate(self) -> None:
        ranges = [
            set(range(self.audio_size)),
            {self.bos_token_id},
            {self.id_token_id},
            set(self.digit_token_ids),
            {self.eos_token_id},
        ]
        for index, left in enumerate(ranges):
            for right in ranges[index + 1 :]:
                if left & right:
                    raise ValueError("Audio, special, and digit token ranges overlap")
        if self.size > torch.iinfo(torch.uint16).max + 1:
            raise ValueError("Vocabulary does not fit uint16 token storage")
