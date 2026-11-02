"""Breakers for a generation that can never stop on its own.

A logits row that is all zero, all -inf, or carries a NaN samples token 0 every
time: ``torch.argmax`` returns index 0 for such a row, and the inverse-CDF draw
leaves its zero-filled ``out`` untouched because no cumulative lane passes the
target. Token 0 is "!" in the Qwen vocabularies and is not an EOS, so the
request decodes it until its output budget runs out with every TP rank pinned
at 100% -- the server is alive but answers nothing. Both faults are failed
loudly instead, the way ``CpuMoeExecutor.raise_if_unhealthy`` surfaces a dead
coordinator rather than serving stale experts.
"""

from __future__ import annotations

import json
import os
import re
from typing import NamedTuple, Protocol

import torch

_LAYER_NAME = re.compile(r"(?:^|\.)layers\.\d+$")


class RowFault(NamedTuple):
    """What was wrong with one logits row, counted before it is failed.

    The split between NaN and +/-Inf is what names the producing stage: a NaN says an
    invalid arithmetic operation happened (a 0/0 softmax normalizer, an accumulator
    seeded from uninitialized memory), an Inf says a value overflowed its dtype. The
    counts are taken on the device for every row of every step and copied to the host
    beside the sampled tokens, so the drain reads them without a synchronization and
    while they still describe the step that produced them.
    """

    n_nan: int
    n_posinf: int
    n_neginf: int
    n_finite: int

    @property
    def n_nonfinite(self) -> int:
        return self.n_nan + self.n_posinf + self.n_neginf

    @property
    def n_total(self) -> int:
        return self.n_nonfinite + self.n_finite

    @property
    def is_fault(self) -> bool:
        return self.n_nonfinite > 0

    @property
    def all_nonfinite(self) -> bool:
        return self.n_finite == 0 and self.n_nonfinite > 0

    def describe(self) -> str:
        kinds = ", ".join(
            f"{n} {name}"
            for name, n in (("NaN", self.n_nan), ("+Inf", self.n_posinf), ("-Inf", self.n_neginf))
            if n
        )
        return (
            f"{kinds or 'no non-finite'} of {self.n_total} logits, "
            f"{self.n_finite} finite"
            + (", row wholly non-finite" if self.all_nonfinite else "")
        )


def row_fault_counts(logits: torch.Tensor) -> torch.Tensor:
    """int32 ``(batch, 4)`` per-row counts of NaN, +Inf, -Inf and finite logits.

    Four reductions over the batch's logits -- a fraction of a millisecond a step and no
    host synchronization -- which is what lets the breaker stay on by default.
    """
    return torch.stack(
        [
            torch.isnan(logits).sum(dim=-1),
            torch.isposinf(logits).sum(dim=-1),
            torch.isneginf(logits).sum(dim=-1),
            torch.isfinite(logits).sum(dim=-1),
        ],
        dim=-1,
    ).to(torch.int32)


def row_fault(counts_row: torch.Tensor) -> RowFault | None:
    """Rebuild one row's :class:`RowFault`; None if the counts cannot be read.

    Diagnostics must never be able to take serving down, so a failure to interpret the
    counts degrades to a missing description instead of an exception out of the drain.
    """
    try:
        n_nan, n_posinf, n_neginf, n_finite = (int(c) for c in counts_row.reshape(-1))
        return RowFault(n_nan, n_posinf, n_neginf, n_finite)
    except Exception:
        return None


def record_storm_event(path: str, payload: dict) -> bool:
    """Append one fault record to ``path``; report whether it landed.

    The scheduler logs to a terminal, and a pane's scrollback is not evidence: the run
    that first caught a non-finite row lost its report to an ``AttributeError`` raised by
    the logging call itself. Writing the record out first means the finding survives
    whatever the logger does afterwards.
    """
    if not path:
        return False
    try:
        expanded = os.path.expanduser(path)
        os.makedirs(os.path.dirname(expanded) or ".", exist_ok=True)
        with open(expanded, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        return True
    except Exception:
        return False


class RepeatState(Protocol):
    repeat_token: int
    repeat_run: int


def note_repeated_token(req: RepeatState, token: int, limit: int) -> bool:
    """Count a run of identical tokens; True once the run exceeds ``limit`` (<=0 disables).

    The backstop for a collapse that stays finite: nothing in the sampler can tell
    a stuck row from a model that genuinely wants ten thousand more exclamation marks.
    """
    if token == req.repeat_token:
        req.repeat_run += 1
    else:
        req.repeat_token = token
        req.repeat_run = 1
    return limit > 0 and req.repeat_run > limit


class NonFiniteTracer:
    """Names the first module whose output went non-finite in the step just run.

    Hooks every ``...layers.<N>`` module in definition order and keeps the earliest bad
    hook id in a device tensor, so a traced step costs one reduction per layer and no
    host sync; it is read only when a row is already being failed. Decode runs inside a
    CUDA graph, whose replay does not re-enter Python, so tracing decode needs
    ``--cuda-graph-max-bs 0``.
    """

    def __init__(self, model: object, device: torch.device) -> None:
        self.marker = torch.full((), -1, dtype=torch.int32, device=device)
        self.names: list[str] = []
        for name, module in model.named_modules():
            if not _LAYER_NAME.search(name):
                continue
            self.names.append(name)
            module.register_forward_hook(self._make_hook(len(self.names) - 1))

    def _make_hook(self, order: int):
        rank = torch.full((), order, dtype=torch.int32, device=self.marker.device)

        def hook(_module, _inputs, output):
            out = output[0] if isinstance(output, (tuple, list)) and output else output
            if not torch.is_tensor(out):
                return
            bad = ~torch.isfinite(out).all()
            # Only write while unset (-1): hooks run in order, so the first bad wins.
            self.marker.copy_(torch.where(bad & (self.marker < 0), rank, self.marker))

        return hook

    def begin_step(self) -> None:
        self.marker.fill_(-1)

    def describe(self) -> str:
        """Where the fault first showed up, phrased for a log line."""
        idx = int(self.marker.item())
        if not 0 <= idx < len(self.names):
            return "no layer module (decode may have run inside a CUDA graph)"
        return self.names[idx]
