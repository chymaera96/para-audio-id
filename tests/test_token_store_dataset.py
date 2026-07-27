import json

import numpy as np
import pytest

from para_audio_id.audio_lm.dataset import (
    AudioTokenDataset,
    CataloguePassBatchSampler,
    collate_causal_documents,
)
from para_audio_id.audio_lm.token_store import (
    TokenRecord,
    TokenStoreIndex,
    validate_shard,
    write_shard,
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
