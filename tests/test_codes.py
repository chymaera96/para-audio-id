import torch

from para_audio_id.codes import (
    BOS_TOKEN,
    assign_codes,
    code_to_tokens,
    teacher_forcing_inputs,
    tokens_to_code,
)


def test_assignment_is_complete_unique_and_deterministic():
    first = assign_codes(100_000, seed=7)
    second = assign_codes(100_000, seed=7)
    assert first == second
    assert len(set(first)) == 100_000
    assert all(len(code) == 5 for code in first)


def test_token_round_trip_preserves_leading_zeroes():
    tokens = code_to_tokens("00109")
    assert tokens_to_code(tokens) == "00109"
    targets = torch.stack((tokens, code_to_tokens("99999")))
    inputs = teacher_forcing_inputs(targets)
    assert inputs[:, 0].tolist() == [BOS_TOKEN, BOS_TOKEN]
    assert torch.equal(inputs[:, 1:], targets[:, :-1])
