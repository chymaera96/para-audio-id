from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
import math

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .token_store import TokenStoreIndex
from .vocabulary import AudioLMVocabulary


class AudioTokenDataset(Dataset):
    def __init__(self, store: TokenStoreIndex, *, expected_segments_per_track: int = 6):
        self.store = store
        available: dict[str, list] = defaultdict(list)
        for record in store.records:
            available[record.track_id].append(record)
        complete_tracks = {
            track_id
            for track_id, records in available.items()
            if len(records) == expected_segments_per_track
        }
        self.dropped_incomplete_tracks = sorted(set(available) - complete_tracks)
        self.records = [
            record for record in store.records if record.track_id in complete_tracks
        ]
        self.by_track: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            self.by_track[record.track_id].append(index)
        for indices in self.by_track.values():
            indices.sort(key=lambda index: self.records[index].segment_start)
        self.track_ids = sorted(self.by_track)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        return {
            "audio_tokens": torch.from_numpy(self.store.tokens(record)),
            "code": record.code,
            "track_id": record.track_id,
            "segment_start": record.segment_start,
            "document_index": record.document_index,
        }


class CataloguePassBatchSampler(Sampler[list[int]]):
    """Seeded batches that expose every cached segment exactly once per pass."""

    def __init__(
        self,
        dataset: AudioTokenDataset,
        *,
        tracks_per_microbatch: int,
        segments_per_track: int,
        seed: int,
        catalogue_pass: int = 0,
        world_size: int = 1,
        rank: int = 0,
    ):
        if tracks_per_microbatch < 1 or segments_per_track < 1:
            raise ValueError("Batch dimensions must be positive")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid distributed rank")
        if tracks_per_microbatch % world_size:
            raise ValueError("tracks_per_microbatch must divide evenly across ranks")
        lengths = {len(indices) for indices in dataset.by_track.values()}
        if any(length % segments_per_track for length in lengths):
            raise ValueError(
                "Every track must have a segment count divisible by segments_per_track"
            )
        self.dataset = dataset
        self.tracks_per_microbatch = tracks_per_microbatch
        self.segments_per_track = segments_per_track
        self.seed = seed
        self.catalogue_pass = catalogue_pass
        self.world_size = world_size
        self.rank = rank
        self.rounds = max(length // segments_per_track for length in lengths)

    def set_epoch(self, catalogue_pass: int) -> None:
        self.catalogue_pass = catalogue_pass

    def __len__(self) -> int:
        return self.rounds * math.ceil(
            len(self.dataset.track_ids) / self.tracks_per_microbatch
        )

    def __iter__(self) -> Iterator[list[int]]:
        local_tracks = self.tracks_per_microbatch // self.world_size
        for segment_round in range(self.rounds):
            track_ids = np.asarray(self.dataset.track_ids, dtype=object)
            rng = np.random.default_rng(
                self.seed + self.catalogue_pass * self.rounds + segment_round
            )
            rng.shuffle(track_ids)
            missing = (-len(track_ids)) % self.tracks_per_microbatch
            if missing:
                track_ids = np.concatenate((track_ids, track_ids[:missing]))
            for offset in range(0, len(track_ids), self.tracks_per_microbatch):
                global_tracks = track_ids[offset : offset + self.tracks_per_microbatch]
                selected = global_tracks[
                    self.rank * local_tracks : (self.rank + 1) * local_tracks
                ]
                batch = []
                for track_id in selected:
                    indices = self.dataset.by_track[str(track_id)]
                    start = segment_round * self.segments_per_track
                    batch.extend(indices[start : start + self.segments_per_track])
                yield batch


def collate_causal_documents(
    examples: list[dict], vocabulary: AudioLMVocabulary, max_positions: int
) -> dict:
    sequences = []
    audio_target_masks = []
    id_target_masks = []
    boundary_target_masks = []
    for example in examples:
        audio = example["audio_tokens"].long()
        digits = vocabulary.encode_code(example["code"])
        sequence = torch.cat(
            (
                torch.tensor([vocabulary.bos_token_id]),
                audio,
                torch.tensor([vocabulary.id_token_id]),
                digits,
                torch.tensor([vocabulary.eos_token_id]),
            )
        )
        if len(sequence) > max_positions:
            raise ValueError(
                f"Document length {len(sequence)} exceeds max_position_embeddings={max_positions}"
            )
        # Masks describe next-token labels after the causal shift.
        audio_mask = torch.zeros(len(sequence) - 1, dtype=torch.bool)
        audio_mask[: len(audio)] = True
        id_mask = torch.zeros(len(sequence) - 1, dtype=torch.bool)
        id_mask[-6:-1] = True
        boundary_mask = torch.zeros(len(sequence) - 1, dtype=torch.bool)
        boundary_mask[len(audio)] = True  # [ID]
        boundary_mask[-1] = True  # [EOS]
        sequences.append(sequence)
        audio_target_masks.append(audio_mask)
        id_target_masks.append(id_mask)
        boundary_target_masks.append(boundary_mask)

    maximum = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), maximum), vocabulary.bos_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros((len(sequences), maximum), dtype=torch.long)
    audio_mask = torch.zeros((len(sequences), maximum - 1), dtype=torch.bool)
    id_mask = torch.zeros((len(sequences), maximum - 1), dtype=torch.bool)
    boundary_mask = torch.zeros((len(sequences), maximum - 1), dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        input_ids[row, : len(sequence)] = sequence
        attention_mask[row, : len(sequence)] = 1
        audio_mask[row, : len(sequence) - 1] = audio_target_masks[row]
        id_mask[row, : len(sequence) - 1] = id_target_masks[row]
        boundary_mask[row, : len(sequence) - 1] = boundary_target_masks[row]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "audio_target_mask": audio_mask,
        "id_target_mask": id_mask,
        "boundary_target_mask": boundary_mask,
        "code": [example["code"] for example in examples],
        "track_id": [example["track_id"] for example in examples],
        "document_index": torch.tensor(
            [example["document_index"] for example in examples], dtype=torch.long
        ),
    }
