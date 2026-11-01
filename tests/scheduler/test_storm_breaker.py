"""The degenerate-generation breakers (freetoken/storm.py) trip on a stuck row and
stay quiet on real output, so a NaN never reaches the client as an endless "!"."""

import torch
import torch.nn as nn

from freetoken.core import Req, SamplingParams
from freetoken.storm import NonFiniteTracer, nonfinite_rows, note_repeated_token


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


class _Blk(nn.Module):
    def __init__(self, poison: bool) -> None:
        super().__init__()
        self.poison = poison

    def forward(self, x):
        return torch.full_like(x, float("nan")) if self.poison else x * 2.0


class _Net(nn.Module):
    def __init__(self, poison_at: int | None) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_Blk(i == poison_at) for i in range(3)]
        )

    def forward(self, x):
        for blk in self.layers:
            x = blk(x)
        return x


def test_tracer_names_the_first_layer_that_went_nonfinite():
    net, x = _Net(poison_at=2), torch.ones(4)
    net.layers[1] = _Blk(True)  # earliest fault must win
    tracer = NonFiniteTracer(net, torch.device("cpu"))
    assert tracer.names == ["layers.0", "layers.1", "layers.2"]
    tracer.begin_step()
    net(x)
    assert tracer.describe() == "layers.1"


def test_tracer_reports_nothing_after_a_clean_step():
    tracer = NonFiniteTracer(_Net(poison_at=None), torch.device("cpu"))
    tracer.begin_step()
    _Net(poison_at=None)(torch.ones(4))
    assert "no layer module" in tracer.describe()


def test_tracer_marker_is_clear_before_a_step_runs():
    tracer = NonFiniteTracer(_Net(poison_at=0), torch.device("cpu"))
    assert "no layer module" in tracer.describe()
    tracer.begin_step()
    assert int(tracer.marker) == -1
