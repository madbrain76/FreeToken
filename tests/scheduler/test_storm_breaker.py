"""The degenerate-generation breakers (freetoken/storm.py) trip on a stuck row and
stay quiet on real output, so a NaN never reaches the client as an endless "!"."""

import torch

from freetoken.core import Req, SamplingParams
from freetoken.storm import nonfinite_rows, note_repeated_token


def _req():
    return Req(
        input_ids=torch.zeros(4, dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=8,
        uid=0,
        sampling_params=SamplingParams(),
        cache_handle=None,
    )


def test_nonfinite_rows_flags_nan_and_inf_only():
    logits = torch.zeros(3, 64)
    logits[1, 5] = float("nan")
    logits[2, 9] = float("inf")
    assert nonfinite_rows(logits).tolist() == [False, True, True]


def test_a_degenerate_row_samples_the_token_the_breaker_exists_for():
    # argmax returns index 0 for every one of these shapes, and index 0 is "!" in the
    # Qwen vocabularies -- not an EOS, so the request cannot stop by itself.
    for row in (
        torch.zeros(64),
        torch.full((64,), float("nan")),
        torch.full((64,), float("-inf")),
    ):
        assert int(torch.argmax(row)) == 0


def test_run_breaker_trips_only_past_the_limit():
    req = _req()
    for _ in range(4):
        assert not note_repeated_token(req, 7, limit=4)
    assert note_repeated_token(req, 7, limit=4)


def test_run_breaker_resets_on_a_different_token():
    req = _req()
    for _ in range(4):
        note_repeated_token(req, 7, limit=4)
    assert not note_repeated_token(req, 8, limit=4)
    assert (req.repeat_token, req.repeat_run) == (8, 1)


def test_run_breaker_is_disabled_at_a_non_positive_limit():
    req = _req()
    for _ in range(1000):
        assert not note_repeated_token(req, 7, limit=0)


def test_req_carries_fresh_breaker_state():
    req = _req()
    assert (req.repeat_token, req.repeat_run) == (-1, 0)
    assert _req() is not req
