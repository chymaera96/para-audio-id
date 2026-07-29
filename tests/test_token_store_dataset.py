import json

import numpy as np
import pytest
import torch

from para_audio_id.audio_lm.dataset import (
    AudioTokenDataset,
    CataloguePassBatchSampler,
    PairedAudioTokenDataset,
    PairedViewBatchSampler,
    collate_causal_documents,
)
from para_audio_id.audio_lm.token_store import (
    TokenRecord,
    TokenStoreIndex,
    validate_shard,
    write_shard,
)
from para_audio_id.audio_lm.training import (
    prefetched_waveforms_for_step,
    replace_secondary_rows_with_noisy_tokens,
)
from para_audio_id.audio_lm.vocabulary import AudioLMVocabulary


def make_store(tmp_path, tracks=4):
    records = []
    parts = []
    offset = 0
    document_index = 0
    for track in range(tracks):
        for segment in range(6):
            tokens = np.array([segment, 1024 + segment], dtype=np.uint16)
            parts.append(tokens)
            records.append(
                TokenRecord(
                    document_index=document_index,
                    track_id=f"track-{track}",
                    code=f"{track:05d}",
                    source_path=f"{track}.mp3",
                    segment_start=segment * 5.0,
                    segment_duration=5.0,
                    status="ok",
                    token_offset=offset,
                    token_count=2,
                    frames=1,
                )
            )
            offset += 2
            document_index += 1
    write_shard(
        tmp_path,
        0,
        records=records,
        tokens=np.concatenate(parts),
        tokenizer_spec={"name": "dummy"},
        tokenizer_fingerprint="fingerprint",
    )
    return TokenStoreIndex(tmp_path, tokenizer_fingerprint="fingerprint")


def make_role_store(root, starts, *, tracks=4, role, view_type):
    records = []
    parts = []
    offset = 0
    for track in range(tracks):
        for start in starts:
            tokens = np.array([int(start), 1024 + int(start)], dtype=np.uint16)
            parts.append(tokens)
            records.append(
                TokenRecord(
                    document_index=len(records),
                    track_id=f"track-{track}",
                    code=f"{track:05d}",
                    source_path=f"{track}.mp3",
                    segment_start=float(start),
                    segment_duration=5.0,
                    status="ok",
                    token_offset=offset,
                    token_count=2,
                    frames=1,
                    view_type=view_type,
                    corpus_role=role,
                )
            )
            offset += 2
    write_shard(
        root,
        0,
        records=records,
        tokens=np.concatenate(parts),
        tokenizer_spec={"name": "dummy"},
        tokenizer_fingerprint="fingerprint",
        corpus_role=role,
    )
    return TokenStoreIndex(
        root, tokenizer_fingerprint="fingerprint", corpus_role=role
    )


def test_manifest_spans_and_fingerprint_validation(tmp_path):
    make_store(tmp_path)
    metadata = validate_shard(tmp_path, 0, tokenizer_fingerprint="fingerprint")
    assert metadata["documents"] == 24
    with pytest.raises(ValueError, match="fingerprint"):
        validate_shard(tmp_path, 0, tokenizer_fingerprint="wrong")

    index_path = tmp_path / "shard-000000.index.jsonl"
    rows = [json.loads(line) for line in index_path.read_text().splitlines()]
    rows[0]["token_offset"] = 10_000
    index_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="out-of-range"):
        validate_shard(tmp_path, 0, tokenizer_fingerprint="fingerprint")


def test_corpus_role_prevents_evaluation_store_from_training(tmp_path):
    make_role_store(
        tmp_path,
        [2.5],
        tracks=1,
        role="heldout_evaluation",
        view_type="heldout",
    )
    with pytest.raises(ValueError, match="corpus role"):
        TokenStoreIndex(
            tmp_path,
            tokenizer_fingerprint="fingerprint",
            corpus_role="shifted_training",
        )


def test_shifted_cache_metadata_records_crop_and_padding(tmp_path):
    record = TokenRecord(
        document_index=0,
        track_id="track",
        code="00000",
        source_path="track.mp3",
        segment_start=24.0,
        segment_duration=5.0,
        status="ok",
        token_offset=0,
        token_count=2,
        frames=1,
        view_type="shifted",
        corpus_role="shifted_training",
        source_duration=26.0,
        padded_samples=72_000,
    )
    write_shard(
        tmp_path,
        0,
        records=[record],
        tokens=np.array([1, 1025], dtype=np.uint16),
        tokenizer_spec={"selected_codebooks": 2, "codebook_size": 1024},
        tokenizer_fingerprint="fingerprint",
        corpus_role="shifted_training",
    )
    row = json.loads(
        (tmp_path / "shard-000000.index.jsonl").read_text().strip()
    )
    assert row["track_id"] == "track"
    assert row["segment_start"] == 24.0
    assert row["segment_duration"] == 5.0
    assert row["source_duration"] == 26.0
    assert row["padded_samples"] == 72_000
    assert row["corpus_role"] == "shifted_training"


def test_sampler_is_deterministic_across_resume(tmp_path):
    dataset = AudioTokenDataset(make_store(tmp_path))
    first = CataloguePassBatchSampler(
        dataset,
        tracks_per_microbatch=4,
        segments_per_track=2,
        seed=7,
        catalogue_pass=3,
    )
    resumed = CataloguePassBatchSampler(
        dataset,
        tracks_per_microbatch=4,
        segments_per_track=2,
        seed=7,
        catalogue_pass=3,
    )
    assert list(first) == list(resumed)
    assert len([index for batch in first for index in batch]) == len(dataset)


def test_training_track_subset_is_seeded_and_complete(tmp_path):
    store = make_store(tmp_path, tracks=10)
    first = AudioTokenDataset(store, max_tracks=4, subset_seed=17)
    repeated = AudioTokenDataset(store, max_tracks=4, subset_seed=17)
    different = AudioTokenDataset(store, max_tracks=4, subset_seed=18)
    assert first.track_ids == repeated.track_ids
    assert first.track_ids != different.track_ids
    assert len(first.track_ids) == 4
    assert len(first) == 4 * 6
    assert first.complete_track_count == 10
    assert len(first.excluded_by_subset) == 6


def test_paired_sampler_has_one_canonical_and_shifted_view_per_track(tmp_path):
    canonical_starts = [0, 5, 10, 15, 20, 25]
    shifted_starts = [1, 2, 3, 4]
    canonical = make_role_store(
        tmp_path / "canonical",
        canonical_starts,
        role="canonical_training",
        view_type="canonical",
    )
    shifted = make_role_store(
        tmp_path / "shifted",
        shifted_starts,
        role="shifted_training",
        view_type="shifted",
    )
    track_ids = [f"track-{track}" for track in range(4)]
    dataset = PairedAudioTokenDataset(
        canonical,
        shifted,
        track_ids=track_ids,
        canonical_starts=canonical_starts,
        shifted_starts=shifted_starts,
        view_mode="paired",
    )
    sampler = PairedViewBatchSampler(
        dataset, tracks_per_microbatch=4, seed=9, catalogue_pass=2
    )
    batch = [dataset[index] for index in next(iter(sampler))]
    assert len(batch) == 8
    for track_id in track_ids:
        views = [row for row in batch if row["track_id"] == track_id]
        assert {row["view_type"] for row in views} == {"canonical", "shifted"}
        assert len({row["code"] for row in views}) == 1
    repeated = PairedViewBatchSampler(
        dataset, tracks_per_microbatch=4, seed=9, catalogue_pass=2
    )
    assert list(sampler) == list(repeated)
    canonical_seen = set()
    shifted_seen = set()
    for catalogue_pass in range(20):
        pass_sampler = PairedViewBatchSampler(
            dataset,
            tracks_per_microbatch=4,
            seed=9,
            catalogue_pass=catalogue_pass,
        )
        rows = [
            dataset[index]
            for batch_indices in pass_sampler
            for index in batch_indices
            if dataset[index]["track_id"] == "track-0"
        ]
        canonical_seen.update(
            row["segment_start"] for row in rows if row["view_type"] == "canonical"
        )
        shifted_seen.update(
            row["segment_start"] for row in rows if row["view_type"] == "shifted"
        )
    assert canonical_seen == set(canonical_starts)
    assert shifted_seen == set(shifted_starts)


def test_canonical_only_sampler_preserves_two_view_batch_shape(tmp_path):
    starts = [0, 5, 10, 15, 20, 25]
    canonical = make_role_store(
        tmp_path / "canonical",
        starts,
        role="canonical_training",
        view_type="canonical",
    )
    dataset = PairedAudioTokenDataset(
        canonical,
        None,
        track_ids=[f"track-{track}" for track in range(4)],
        canonical_starts=starts,
        shifted_starts=[],
        view_mode="canonical_only",
    )
    batch = [
        dataset[index]
        for index in next(
            iter(PairedViewBatchSampler(dataset, tracks_per_microbatch=4, seed=3))
        )
    ]
    assert len(batch) == 8
    for track_id in dataset.track_ids:
        views = [row for row in batch if row["track_id"] == track_id]
        assert len(views) == 2
        assert all(row["view_type"] == "canonical" for row in views)
        assert len({row["segment_start"] for row in views}) == 2


def test_causal_document_masks_and_no_first_digit_leakage(tmp_path):
    dataset = AudioTokenDataset(make_store(tmp_path, tracks=1))
    example = dataset[0]
    vocabulary = AudioLMVocabulary()
    batch = collate_causal_documents([example], vocabulary, 32)
    sequence = batch["input_ids"][0]
    id_position = int((sequence == vocabulary.id_token_id).nonzero()[0])
    first_digit_position = id_position + 1
    assert sequence[first_digit_position] == vocabulary.encode_code(example["code"])[0]
    assert sequence[:first_digit_position].tolist() == [
        vocabulary.bos_token_id,
        *example["audio_tokens"].tolist(),
        vocabulary.id_token_id,
    ]
    assert batch["audio_target_mask"].sum() == len(example["audio_tokens"])
    assert batch["id_target_mask"].sum() == 5
    assert batch["id_target_mask"][0, id_position]
    assert sequence[-1] == vocabulary.eos_token_id
    assert batch["boundary_target_mask"].sum() == 2
    assert batch["boundary_target_mask"][0, id_position - 1]
    assert batch["boundary_target_mask"][0, -1]
    assert not (
        batch["audio_target_mask"]
        & batch["id_target_mask"]
        | batch["audio_target_mask"]
        & batch["boundary_target_mask"]
        | batch["id_target_mask"]
        & batch["boundary_target_mask"]
    ).any()


def test_noisy_secondary_is_an_exact_copy_of_the_clean_anchor():
    vocabulary = AudioLMVocabulary()
    examples = [
        {
            "audio_tokens": torch.tensor([index + 1, 1025 + index]),
            "code": f"{pair:05d}",
            "track_id": f"track-{pair}",
            "source_path": f"{pair}.mp3",
            "segment_start": float(index),
            "segment_duration": 5.0,
            "document_index": index,
            "view_type": "canonical" if index % 2 == 0 else "shifted",
        }
        for pair in range(2)
        for index in (pair * 2, pair * 2 + 1)
    ]
    batch = collate_causal_documents(examples, vocabulary, 32)
    replaced = replace_secondary_rows_with_noisy_tokens(
        batch,
        torch.tensor([[77, 1101]]),
        [0],
        id_token_id=vocabulary.id_token_id,
    )
    assert replaced["is_noisy"].tolist() == [False, True, False, False]
    assert replaced["view_type"][1] == "noisy"
    for field in (
        "track_id",
        "code",
        "source_path",
    ):
        assert replaced[field][1] == replaced[field][0]
    for field in ("segment_start", "segment_duration", "document_index"):
        assert replaced[field][1] == replaced[field][0]
    id_column = int(
        (replaced["input_ids"][0] == vocabulary.id_token_id).nonzero()[0]
    )
    assert replaced["input_ids"][1, 1:id_column].tolist() == [77, 1101]
    assert torch.equal(
        replaced["input_ids"][1, id_column:],
        replaced["input_ids"][0, id_column:],
    )
    assert replaced["input_ids"][2:].equal(batch["input_ids"][2:])


def test_prefetched_waveforms_cannot_override_actual_sampler_step():
    anchor = torch.ones(1, 8)
    noise = torch.zeros(1, 8)
    batch = {
        "planned_optimizer_step": 20,
        "planned_batch_idx": 3,
        "anchor_waveforms": anchor,
        "noise_waveforms": noise,
        "loaded_pair_indices": [2],
    }
    selected = prefetched_waveforms_for_step(
        batch, global_step=20, batch_idx=3
    )
    assert set(selected) == {2}
    assert torch.equal(selected[2][0], anchor[0])
    assert torch.equal(selected[2][1], noise[0])
    assert (
        prefetched_waveforms_for_step(
            batch, global_step=21, batch_idx=3
        )
        == {}
    )
