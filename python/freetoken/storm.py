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

import re
from typing import Protocol

import torch

_LAYER_NAME = re.compile(r"(?:^|\.)layers\.\d+$")


def nonfinite_rows(logits: torch.Tensor) -> torch.Tensor:
    """Bool mask over the batch: rows holding at least one NaN or Inf logit."""
    return ~torch.isfinite(logits).all(dim=-1)


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
