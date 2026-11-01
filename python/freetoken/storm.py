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

from typing import Protocol

import torch


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
