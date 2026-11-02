"""The degenerate-generation breakers (freetoken/storm.py) trip on a stuck row and
stay quiet on real output, so a NaN never reaches the client as an endless "!"."""

import json
import pathlib
import re
from types import SimpleNamespace

import torch
import torch.nn as nn

from freetoken.core import Batch, Req, SamplingParams
from freetoken.scheduler.scheduler import Scheduler
from freetoken.storm import (
    NonFiniteTracer,
    note_repeated_token,
    row_fault,
    row_fault_counts,
)


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


def test_row_fault_counts_separate_nan_from_inf():
    logits = torch.zeros(3, 64)
    logits[1, 5] = float("nan")
    logits[2, 9] = float("inf")
    logits[2, 10] = float("-inf")
    counts = row_fault_counts(logits)
    assert counts.dtype == torch.int32
    assert counts.tolist() == [[0, 0, 0, 64], [1, 0, 0, 63], [0, 1, 1, 62]]
    assert not row_fault(counts[0]).is_fault
    assert row_fault(counts[1]).is_fault and not row_fault(counts[1]).all_nonfinite
    wholly = row_fault(row_fault_counts(torch.full((1, 8), float("nan")))[0])
    assert wholly.is_fault and wholly.all_nonfinite
    assert "wholly non-finite" in wholly.describe()


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



# ---------------------------------------------------------------------------
# The drain itself. The first version of the breaker reported its trip with a
# logger method that did not exist, so the AttributeError escaped
# Scheduler._process_last_data, killed every scheduler rank and took the API
# server with it: a fault report that costs more than the request it reports.
# These tests run the real drain, so they fail if it can ever raise again.


class _DecodeManager:
    def __init__(self):
        self.running_reqs = []
        self.removed = []

    def remove_req(self, req):
        self.removed.append(req.uid)


class _CacheManager:
    class _Region:
        def __enter__(self):
            return None

        def __exit__(self, *_):
            return False

    def lazy_free_region(self):
        return self._Region()


class _Reporter:
    def report_batch(self, *args, **kwargs):
        return None


class _StubScheduler:
    """Exactly the attributes ``_process_last_data`` touches."""

    # The real reporter, not a stand-in: whether a trip can raise is the thing under test.
    _degenerate_generation_reply = Scheduler._degenerate_generation_reply

    def __init__(self):
        self.config = SimpleNamespace(step_timeout=120.0, page_size=64)
        self.engine = SimpleNamespace(nan_tracer=None)
        self.cache_manager = _CacheManager()
        self.decode_manager = _DecodeManager()
        self.prefill_manager = SimpleNamespace(pending_list=[])
        self.status_reporter = _Reporter()
        self.finished_reqs = set()
        self.eos_token_ids = {248046}
        self.toolcall_anchor_id = -1
        self.freed = []
        self.sent = []

    def _wait_copy_done(self, copy_done, batch, timeout=None):
        return None

    def _free_req_resources(self, req):
        self.freed.append(req.uid)

    def _kv_usage_pages(self):
        return (0, 0)

    def _mamba_slot_usage(self):
        return (0, 0)

    def _swa_token_usage(self):
        return (0, 0)

    def _gpu_mem_bytes(self):
        return 0

    def _match_stop_str(self, req):
        return None

    def send_result(self, msgs):
        self.sent.append(list(msgs))


def _drain(stub, reqs, tokens, counts=None, phase="decode"):
    batch = Batch(reqs=reqs, phase=phase)
    batch.padded_reqs = list(reqs)
    forward_output = (
        None,
        torch.tensor(tokens, dtype=torch.int32),
        None,
        None if counts is None else torch.tensor(counts, dtype=torch.int32),
        None,
    )
    Scheduler._process_last_data(stub, (SimpleNamespace(batch=batch), forward_output))
    return batch


def _replies(stub):
    return [m for sent in stub.sent for m in sent]


def _errors(stub, code):
    return [m for m in _replies(stub) if getattr(m, "code", None) == code]


def test_drain_ships_tokens_for_a_healthy_batch():
    stub = _StubScheduler()
    _drain(stub, [_req()], [7], [[0, 0, 0, 64]])
    assert [(m.uid, m.next_token, m.finished) for m in _replies(stub)] == [(0, 7, False)]
    assert stub.freed == [] and stub.decode_manager.removed == []


def test_drain_fails_a_nonfinite_row_instead_of_streaming_it():
    stub = _StubScheduler()
    healthy, poisoned = _req(), _req()
    healthy.uid, poisoned.uid = 10, 11
    _drain(stub, [healthy, poisoned], [7, 0], [[0, 0, 0, 64], [4, 0, 1, 59]])
    errors = _errors(stub, "nonfinite_logits")
    assert [m.uid for m in errors] == [11]
    assert "4 NaN" in errors[0].error and "-Inf" in errors[0].error
    shipped = [m for m in _replies(stub) if hasattr(m, "next_token")]
    assert [(m.uid, m.next_token) for m in shipped] == [(10, 7)]
    assert stub.freed == [11] and stub.decode_manager.removed == [11]


def test_drain_survives_a_reporter_that_raises(monkeypatch):
    # The exact failure that took the server down: the reporting call itself blows up.
    class _BrokenLogger:
        def error_rank0(self, *args, **kwargs):
            raise AttributeError("'Logger' object has no attribute 'error_rank0'")

        def warning(self, *args, **kwargs):
            raise RuntimeError("logging is down entirely")

        def warning_rank0(self, *args, **kwargs):
            raise RuntimeError("logging is down entirely")

    monkeypatch.setattr("freetoken.scheduler.scheduler.logger", _BrokenLogger())
    stub = _StubScheduler()
    poisoned = _req()
    poisoned.uid = 12
    _drain(stub, [poisoned], [0], [[64, 0, 0, 0]])
    assert [m.uid for m in _errors(stub, "nonfinite_logits")] == [12]
    assert stub.freed == [12], "the request must still be failed and released"


def test_a_trip_is_recorded_where_a_lost_terminal_cannot_hide_it(monkeypatch, tmp_path):
    from freetoken.env import ENV

    events = tmp_path / "storm-events.jsonl"
    monkeypatch.setattr(ENV.STORM_EVENT_LOG, "value", str(events))
    stub = _StubScheduler()
    poisoned = _req()
    poisoned.uid = 13
    _drain(stub, [poisoned], [0], [[64, 0, 0, 0]], phase="prefill")
    record = json.loads(events.read_text().strip())
    assert record["event"] == "nonfinite_logits" and record["uid"] == 13
    assert record["phase"] == "prefill" and record["padded_batch"] == 1
    assert record["first_nonfinite_module"] == "NAN_TRACE off"


def test_a_broken_event_log_path_cannot_stop_the_breaker(monkeypatch):
    from freetoken.env import ENV

    monkeypatch.setattr(ENV.STORM_EVENT_LOG, "value", "/proc/nope/storm-events.jsonl")
    stub = _StubScheduler()
    poisoned = _req()
    poisoned.uid = 14
    _drain(stub, [poisoned], [0], [[64, 0, 0, 0]])
    assert [m.uid for m in _errors(stub, "nonfinite_logits")] == [14]


def test_repeat_run_breaker_fails_a_stuck_but_finite_generation(monkeypatch):
    from freetoken.env import ENV

    monkeypatch.setattr(ENV.STORM_TOKEN_RUN, "value", 2)
    stub = _StubScheduler()
    req = _req()
    req.uid = 15
    clean = [[0, 0, 0, 64]]
    _drain(stub, [req], [42], clean)
    _drain(stub, [req], [42], clean)
    assert [m for m in _replies(stub) if getattr(m, "code", None)] == []
    _drain(stub, [req], [42], clean)
    errors = _errors(stub, "degenerate_repeat")
    assert [m.uid for m in errors] == [15] and "repeated 3x" in errors[0].error
    assert stub.freed == [15]


def test_every_logger_helper_the_sources_call_exists():
    # The fork shipped logger.error_rank0 call sites against a helper that was never
    # bound, so its step-timeout handler raised AttributeError. The rank0 helpers are
    # attached at runtime, which no type checker can see, so check them here.
    from freetoken.utils.logger import init_logger

    logger = init_logger("test_api_surface")
    root = pathlib.Path(__file__).resolve().parents[2] / "python" / "freetoken"
    called = set()
    for source in root.rglob("*.py"):
        called.update(re.findall(r"\blogger\.([a-z_0-9]+)\(", source.read_text()))
    assert called, "the scan should find logger calls"
    missing = sorted(name for name in called if not hasattr(logger, name))
    assert not missing, f"logger.{missing} called but never bound"
