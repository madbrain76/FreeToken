"""Pure GPU-memory budget policy shared by startup auto-sizing and runtime rebuild.

No torch/GPU side effects: every function here is integer/byte arithmetic over already-
measured quantities, so it is unit-testable without a device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from freetoken.utils import div_ceil

if TYPE_CHECKING:
    import torch


def layer_slot_bytes(sources: dict[str, "list[torch.Tensor]"]) -> list[int]:
    """Bytes one expert slot of each layer occupies on GPU: its row bytes summed over all banks.

    Each bank source is per-layer ``[num_experts, *row_shape]`` tensors and is
    already TP-sharded upstream, so the per-row byte count is the per-rank slot
    size.
    """
    # marlin/b12x gate_up/down alpha scales are fixed [L*E] residency (do not scale
    # with cache_size), so they are intentionally excluded from the per-slot growth term.
    # tensor[0].numel() is the per-row element count (one expert slot); see the matching
    # slot-byte idiom in kvcache/linear_state_pool.py and kvcache/dsv4/v4_pool.py.
    num_layers = len(next(iter(sources.values())))
    return [sum(t[l][0].numel() * t[l].element_size() for t in sources.values()) for l in range(num_layers)]


def expert_bytes_per_slot(sources: dict[str, "list[torch.Tensor]"]) -> int:
    """Bytes one expert slot occupies on GPU, rounded up to the mean over layers when their bank layouts differ."""
    per_layer = layer_slot_bytes(sources)
    return -(-sum(per_layer) // len(per_layer))


def net_cache_budget_bytes(
    memory_ratio: float, baseline_free: int, weights_bytes: int, fixed_cache_size: int
) -> int:
    """Net GPU bytes available for the MoE + KV pools: ``memory_ratio`` of the pre-model
    baseline minus weights and fixed (non-paged) cache. The ``(1-memory_ratio)`` remainder
    is the CUDA-graph/activation headroom. Single source of truth for startup auto-sizing
    and the runtime-rebuild fit check."""
    return int(memory_ratio * baseline_free) - weights_bytes - fixed_cache_size


def required_bytes(
    moe_cache_size: int, num_pages: int, per_expert_bytes: int, cache_per_page: int
) -> int:
    """GPU bytes a ``(moe_cache_size, num_pages)`` geometry occupies (MoE slots + KV pages)."""
    return moe_cache_size * per_expert_bytes + num_pages * cache_per_page


def plan_cache_budget(
    budget_bytes: int,
    per_expert_bytes: int,
    cache_per_page: int,
    num_experts: int,
    total_experts: int,
    prefill_overlap: bool,
    kv_reserve_pages: int,
    max_slots: int,
) -> tuple[int, int, bool]:
    """Split ``budget_bytes`` MoE-first into (moe_cache_size, num_pages, prefill_overlap).

    ``budget_bytes`` is the net pool for MoE cache + KV cache (caller already subtracted
    weights + fixed_cache_size; the (1-memory_ratio) remainder is the graph headroom).
    Experts greedily fill the budget after reserving ``kv_reserve_pages`` for KV, clamped
    to ``[floor, min(total_experts, max_slots)]`` (floor is ``2*num_experts`` when prefill
    overlap is feasible else ``num_experts``); KV pages take whatever remains.
    """
    assert per_expert_bytes > 0, "per_expert_bytes must be positive"
    assert cache_per_page > 0, "cache_per_page must be positive (owned-KV models unsupported here)"

    hi = min(total_experts, max_slots)
    # Prefill overlap borrows two full expert-layer buffers, so it needs >= 2*num_experts
    # slots; disable it (and lower the floor) if the cap cannot fit that.
    overlap = prefill_overlap and hi >= 2 * num_experts
    lo = 2 * num_experts if overlap else num_experts
    assert hi >= lo, f"slot cap {hi} below the minimum {lo} slots"

    kv_reserve_bytes = kv_reserve_pages * cache_per_page
    # MoE-priority: reserve KV first, then experts greedily take the remaining budget.
    raw = (budget_bytes - kv_reserve_bytes) // per_expert_bytes
    moe_cache_size = max(lo, min(raw, hi))
    # A tiny budget may have forced moe_cache_size below 2*num_experts even with overlap on.
    overlap = overlap and moe_cache_size >= 2 * num_experts

    remaining = budget_bytes - moe_cache_size * per_expert_bytes
    num_pages = max(remaining // cache_per_page, kv_reserve_pages)
    # A tiny budget can floor num_pages at kv_reserve_pages even when ``remaining`` is below
    # the reserve (or negative), yielding a plan that exceeds budget_bytes. Reject here so
    # --moe-cache-auto fails in arithmetic instead of OOMing in a later CUDA allocation.
    total = moe_cache_size * per_expert_bytes + num_pages * cache_per_page
    assert total <= budget_bytes, (
        f"cache budget too small: minimum plan (moe={moe_cache_size} slots, "
        f"kv={num_pages} pages) needs {total} B > budget {budget_bytes} B "
        "(raise memory_ratio, lower kv_reserve_tokens, or free GPU memory)"
    )
    assert num_pages > 1, "not enough memory for KV cache after MoE allocation"
    return moe_cache_size, num_pages, overlap


def resolve_moe_cache_auto(
    *,
    baseline_free: int,
    weights_bytes: int,
    memory_ratio: float,
    cache_per_page: int,
    fixed_cache_size: int,
    per_expert_bytes: int,
    num_experts: int,
    total_experts: int,
    prefill_overlap: bool,
    kv_reserve_tokens: int,
    page_size: int,
    max_slots: int | None = None,
) -> tuple[int, int, bool]:
    """Resolve --moe-cache-auto into (moe_cache_size, num_pages, prefill_overlap).

    ``max_slots`` is the expert kernel's addressable slot limit; the plan never exceeds it.

    Applies memory_ratio to the persisted pre-model baseline exactly once, then defers
    the MoE-vs-KV split to plan_cache_budget. The (1-memory_ratio) remainder is the
    CUDA-graph/activation headroom (not subtracted here).
    """
    budget_bytes = net_cache_budget_bytes(memory_ratio, baseline_free, weights_bytes, fixed_cache_size)
    max_slots = total_experts if max_slots is None else min(max_slots, total_experts)
    kv_reserve_pages = div_ceil(kv_reserve_tokens, page_size)
    return plan_cache_budget(
        budget_bytes=budget_bytes,
        per_expert_bytes=per_expert_bytes,
        cache_per_page=cache_per_page,
        num_experts=num_experts,
        total_experts=total_experts,
        prefill_overlap=prefill_overlap,
        kv_reserve_pages=kv_reserve_pages,
        max_slots=max_slots,
    )


# A KV/MoE byte swap at 16450 slots left a 64k prefill ~100 MiB short of the layer-major
# tail. This fixed scratch is that gap, with margin. It does not grow with the prompt:
# 14500 slots plus a 200k KV pool already served long prefill.
PREFILL_HEADROOM_BYTES = 384 * 1024 * 1024


def prefill_scratch_bytes(prompt_tokens: int) -> int:
    """Bytes held back from the expert cache on a grow, so the prefill still fits.

    Same amount at 64k and at 250k. The KV pages themselves pay for the longer context.
    """
    if prompt_tokens <= 0:
        return 0
    return PREFILL_HEADROOM_BYTES


def plan_elastic_kv(
    *,
    budget_bytes: int,
    per_expert_bytes: int,
    cache_per_page: int,
    page_size: int,
    kv_tokens: int,
    num_experts: int,
    total_experts: int,
    prefill_overlap: bool,
    max_slots: int,
    moe_floor: int,
    activation_bytes: int = 0,
) -> tuple[int, int, bool]:
    """Pin the KV pool to ``kv_tokens`` (rounded up to a page) and give the rest of the budget to MoE.

    Leftover bytes stay unused, so the pool is the requested size rather than "whatever remains
    after experts". Raises ``ValueError`` when that KV pin plus ``moe_floor`` does not fit.
    ``prefill_overlap`` stays on only while the resulting slot count can hold two expert layers.
    """
    assert per_expert_bytes > 0 and cache_per_page > 0 and page_size > 0
    hi = min(total_experts, max_slots)
    if moe_floor > hi:
        raise ValueError(f"MoE floor {moe_floor} slots exceeds the slot cap {hi}")
    pages = max(2, div_ceil(kv_tokens, page_size))
    kv_bytes = pages * cache_per_page
    hold = max(0, activation_bytes)
    raw = (budget_bytes - kv_bytes - hold) // per_expert_bytes
    if raw < moe_floor:
        raise ValueError(
            f"elastic KV of {pages * page_size} tokens leaves {raw} expert slots, "
            f"below the floor of {moe_floor}"
        )
    moe_cache_size = min(raw, hi)
    overlap = prefill_overlap and moe_cache_size >= 2 * num_experts
    assert moe_cache_size * per_expert_bytes + kv_bytes + hold <= budget_bytes
    return moe_cache_size, pages, overlap


def max_elastic_tokens(
    *,
    budget_bytes: int,
    per_expert_bytes: int,
    cache_per_page: int,
    page_size: int,
    moe_floor: int,
) -> int:
    """Largest page-aligned KV token count that still leaves ``moe_floor`` expert slots."""
    room = budget_bytes - moe_floor * per_expert_bytes
    if room < 2 * cache_per_page:
        return 0
    return (room // cache_per_page) * page_size


def elastic_grow_tokens(current_tokens: int, needs: list[int], kv_max: int) -> int | None:
    """Pool size that covers the longest prefix of ``needs`` that fits under ``kv_max``.

    Needs above ``kv_max`` are skipped (the caller rejects them). Returns ``None`` when the
    current pool already covers that prefix, so a second request can wait without a rebuild.
    """
    acc = 0
    for need in needs:
        if need > kv_max:
            continue
        if acc + need > kv_max:
            break
        acc += need
    if acc <= current_tokens:
        return None
    return acc


@dataclass(frozen=True)
class ElasticKv:
    """Startup budget shared by every TP rank, so a grow replans without re-reading GPU memory."""

    budget_bytes: int
    per_expert_bytes: int
    cache_per_page: int
    page_size: int
    num_experts: int
    total_experts: int
    max_slots: int
    prefill_overlap: bool
    moe_floor: int
    initial_tokens: int
    max_tokens: int
    # Startup MoE slot count to restore when shrinking back to ``initial_tokens`` (0 -> replan).
    initial_moe_slots: int = 0
    # Kept so a restart still accepts state built with the old field. The grow no longer
    # bills ``tokens * width``; see ``prefill_scratch_bytes``.
    activation_width: int = 0
    activation_dtype_bytes: int = 2

    def plan(self, kv_tokens: int, *, activation_tokens: int = 0) -> tuple[int, int, bool]:
        hold = prefill_scratch_bytes(activation_tokens)
        return plan_elastic_kv(
            budget_bytes=self.budget_bytes,
            per_expert_bytes=self.per_expert_bytes,
            cache_per_page=self.cache_per_page,
            page_size=self.page_size,
            kv_tokens=kv_tokens,
            num_experts=self.num_experts,
            total_experts=self.total_experts,
            prefill_overlap=self.prefill_overlap,
            max_slots=self.max_slots,
            moe_floor=self.moe_floor,
            activation_bytes=hold,
        )

    def shrink_target(self, current_tokens: int) -> tuple[int, int] | None:
        """Return ``(moe_slots, num_pages)`` to restore initial geometry if expanded, else None."""
        if current_tokens <= self.initial_tokens:
            return None
        if self.initial_moe_slots > 0:
            pages = div_ceil(self.initial_tokens, self.page_size)
            return self.initial_moe_slots, pages
        moe, pages, _overlap = self.plan(self.initial_tokens)
        return moe, pages
