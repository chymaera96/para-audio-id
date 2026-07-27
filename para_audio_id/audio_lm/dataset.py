from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .token_store import TokenStoreIndex
from .vocabulary import AudioLMVocabulary


class AudioTokenDataset(Dataset):
    def __init__(
        self,
        store: TokenStoreIndex,
        *,
        expected_segments_per_track: int = 6,
        max_tracks: int | None = None,
        subset_seed: int = 0,
    ):
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
        self.complete_track_count = len(complete_tracks)
        if max_tracks is not None:
            if max_tracks < 1:
                raise ValueError("max_tracks must be positive when provided")
            if max_tracks > len(complete_tracks):
                raise ValueError(
                    f"Requested {max_tracks} training tracks, but only "
                    f"{len(complete_tracks)} have complete token sets"
                )
            candidates = np.asarray(sorted(complete_tracks), dtype=object)
            generator = np.random.default_rng(subset_seed)
            selected = generator.choice(candidates, size=max_tracks, replace=False)
            selected_tracks = {str(track_id) for track_id in selected}
        else:
            selected_tracks = complete_tracks
        self.excluded_by_subset = sorted(complete_tracks - selected_tracks)
        self.records = [
            record for record in store.records if record.track_id in selected_tracks
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
            "view_type": record.view_type,
        }


@dataclass(frozen=True)
class ViewEntry:
    store: TokenStoreIndex
    record_index: int
    view_type: str


def _stable_permutation(length: int, *, seed: int, track_id: str, view: str) -> list[int]:
    digest = hashlib.sha256(f"{seed}:{track_id}:{view}".encode()).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return [int(value) for value in generator.permutation(length)]


class PairedAudioTokenDataset(Dataset):
    def __init__(
        self,
        canonical_store: TokenStoreIndex,
        shifted_store: TokenStoreIndex | None,
        *,
        track_ids: list[str],
        canonical_starts: list[float],
        shifted_starts: list[float],
        view_mode: str,
    ):
        if view_mode not in {"paired", "canonical_only"}:
            raise ValueError(f"Unsupported view_mode {view_mode!r}")
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("Training track IDs are not unique")
        self.view_mode = view_mode
        self.entries: list[ViewEntry] = []
        self.by_track_view: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        selected = set(track_ids)

        def add_store(store: TokenStoreIndex, view_type: str) -> None:
            for record_index, record in enumerate(store.records):
                if record.track_id not in selected:
                    continue
                index = len(self.entries)
                self.entries.append(ViewEntry(store, record_index, view_type))
                self.by_track_view[record.track_id][view_type].append(index)

        add_store(canonical_store, "canonical")
        if view_mode == "paired":
            if shifted_store is None:
                raise ValueError("paired mode requires a shifted-training token store")
            if shifted_store.corpus_role != "shifted_training":
                raise ValueError("Paired training requires a shifted_training store")
            add_store(shifted_store, "shifted")
        expected = {
            "canonical": [float(value) for value in canonical_starts],
            "shifted": [float(value) for value in shifted_starts],
        }
        for track_id in track_ids:
            required_views = ("canonical", "shifted") if view_mode == "paired" else ("canonical",)
            codes = set()
            for view_type in required_views:
                indices = self.by_track_view[track_id][view_type]
                indices.sort(key=lambda index: self._record(index).segment_start)
                actual = [self._record(index).segment_start for index in indices]
                codes.update(self._record(index).code for index in indices)
                if actual != expected[view_type]:
                    raise ValueError(
                        f"Track {track_id} has {view_type} starts {actual}, "
                        f"expected {expected[view_type]}"
                    )
            if len(codes) != 1:
                raise ValueError(f"Track {track_id} has inconsistent identifiers {codes}")
        self.track_ids = list(track_ids)
        self.records = [self._record(index) for index in range(len(self.entries))]

    def _record(self, index: int):
        entry = self.entries[index]
        return entry.store.records[entry.record_index]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict:
        entry = self.entries[index]
        record = entry.store.records[entry.record_index]
        return {
            "audio_tokens": torch.from_numpy(entry.store.tokens(record)),
            "code": record.code,
            "track_id": record.track_id,
            "segment_start": record.segment_start,
            "document_index": record.document_index,
            "view_type": entry.view_type,
        }

    def view_indices(self, track_id: str, view_type: str) -> list[int]:
        return self.by_track_view[track_id][view_type]


class CachedPositionDataset(Dataset):
    def __init__(
        self,
        canonical_store: TokenStoreIndex,
        evaluation_store: TokenStoreIndex,
        *,
        track_ids: list[str],
    ):
        if evaluation_store.corpus_role != "heldout_evaluation":
            raise ValueError("Evaluation dataset requires a heldout_evaluation store")
        selected = set(track_ids)
        self.entries: list[ViewEntry] = []
        self.by_track_start: dict[tuple[str, str, float], int] = {}
        for store, view_type in (
            (canonical_store, "canonical"),
            (evaluation_store, "heldout"),
        ):
            for record_index, record in enumerate(store.records):
                if record.track_id not in selected:
                    continue
                index = len(self.entries)
                self.entries.append(ViewEntry(store, record_index, view_type))
                self.by_track_start[(record.track_id, view_type, record.segment_start)] = index

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict:
        entry = self.entries[index]
        record = entry.store.records[entry.record_index]
        return {
            "audio_tokens": torch.from_numpy(entry.store.tokens(record)),
            "code": record.code,
            "track_id": record.track_id,
            "segment_start": record.segment_start,
            "document_index": record.document_index,
            "view_type": entry.view_type,
        }

    def indices_for(self, track_ids: list[str], view_type: str, start: float) -> list[int]:
        return [self.by_track_start[(track_id, view_type, start)] for track_id in track_ids]


class PairedViewBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: PairedAudioTokenDataset,
        *,
        tracks_per_microbatch: int,
        seed: int,
        catalogue_pass: int = 0,
        world_size: int = 1,
        rank: int = 0,
        batch_count_multiple: int = 1,
    ):
        if tracks_per_microbatch < 1 or tracks_per_microbatch % world_size:
            raise ValueError("tracks_per_microbatch must divide evenly across ranks")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid distributed rank")
        if batch_count_multiple < 1:
            raise ValueError("batch_count_multiple must be positive")
        self.dataset = dataset
        self.tracks_per_microbatch = tracks_per_microbatch
        self.seed = seed
        self.catalogue_pass = catalogue_pass
        self.world_size = world_size
        self.rank = rank
        self.batch_count_multiple = batch_count_multiple

    def set_epoch(self, catalogue_pass: int) -> None:
        self.catalogue_pass = catalogue_pass

    def __len__(self) -> int:
        batches = math.ceil(
            len(self.dataset.track_ids) / self.tracks_per_microbatch
        )
        return math.ceil(batches / self.batch_count_multiple) * self.batch_count_multiple

    def _pair(self, track_id: str) -> list[int]:
        canonical = self.dataset.view_indices(track_id, "canonical")
        canonical_order = _stable_permutation(
            len(canonical), seed=self.seed, track_id=track_id, view="canonical"
        )
        if self.dataset.view_mode == "paired":
            shifted = self.dataset.view_indices(track_id, "shifted")
            shifted_order = _stable_permutation(
                len(shifted), seed=self.seed, track_id=track_id, view="shifted"
            )
            return [
                canonical[canonical_order[self.catalogue_pass % len(canonical)]],
                shifted[shifted_order[self.catalogue_pass % len(shifted)]],
            ]
        offset = (2 * self.catalogue_pass) % len(canonical)
        return [
            canonical[canonical_order[offset]],
            canonical[canonical_order[(offset + 1) % len(canonical)]],
        ]

    def __iter__(self) -> Iterator[list[int]]:
        local_tracks = self.tracks_per_microbatch // self.world_size
        track_ids = np.asarray(self.dataset.track_ids, dtype=object)
        generator = np.random.default_rng(self.seed + self.catalogue_pass)
        generator.shuffle(track_ids)
        missing = (-len(track_ids)) % self.tracks_per_microbatch
        if missing:
            track_ids = np.concatenate((track_ids, track_ids[:missing]))
        first_batches: list[list[int]] = []
        yielded = 0
        for offset in range(0, len(track_ids), self.tracks_per_microbatch):
            global_tracks = track_ids[offset : offset + self.tracks_per_microbatch]
            selected = global_tracks[
                self.rank * local_tracks : (self.rank + 1) * local_tracks
            ]
            batch = [
                index
                for track_id in selected
                for index in self._pair(str(track_id))
            ]
            if len(first_batches) < self.batch_count_multiple:
                first_batches.append(batch)
            yielded += 1
            yield batch
        for index in range(len(self) - yielded):
            yield first_batches[index % len(first_batches)]


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
        batch_count_multiple: int = 1,
    ):
        if tracks_per_microbatch < 1 or segments_per_track < 1:
            raise ValueError("Batch dimensions must be positive")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid distributed rank")
        if tracks_per_microbatch % world_size:
            raise ValueError("tracks_per_microbatch must divide evenly across ranks")
        if batch_count_multiple < 1:
            raise ValueError("batch_count_multiple must be positive")
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
        self.batch_count_multiple = batch_count_multiple
        self.rounds = max(length // segments_per_track for length in lengths)

    def set_epoch(self, catalogue_pass: int) -> None:
        self.catalogue_pass = catalogue_pass

    def __len__(self) -> int:
        batches = self.rounds * math.ceil(
            len(self.dataset.track_ids) / self.tracks_per_microbatch
        )
        return math.ceil(batches / self.batch_count_multiple) * self.batch_count_multiple

    def __iter__(self) -> Iterator[list[int]]:
        local_tracks = self.tracks_per_microbatch // self.world_size
        first_batches: list[list[int]] = []
        yielded = 0
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
                if len(first_batches) < self.batch_count_multiple:
                    first_batches.append(batch)
                yielded += 1
                yield batch
        for index in range(len(self) - yielded):
            yield first_batches[index % len(first_batches)]


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
        "view_type": [example.get("view_type", "canonical") for example in examples],
        "segment_start": torch.tensor(
            [example.get("segment_start", 0.0) for example in examples],
            dtype=torch.float32,
        ),
    }
