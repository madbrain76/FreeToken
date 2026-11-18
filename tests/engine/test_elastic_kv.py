"""Elastic KV: pin the pool at X, grow it toward Y, and give the freed bytes back to MoE."""

from __future__ import annotations

import pytest

from freetoken.engine.cache_budget import (
    ElasticKv,
    elastic_grow_tokens,
    max_elastic_tokens,
    plan_elastic_kv,
)


def _plan(**kwargs):
    base = dict(
        budget_bytes=2_000,
        per_expert_bytes=100,
        cache_per_page=10,
        page_size=10,
        num_experts=4,
        total_experts=50,
        prefill_overlap=True,
        max_slots=50,
        moe_floor=8,
    )
    base.update(kwargs)
    return plan_elastic_kv(**base)


def test_initial_pool_is_the_requested_size_and_moe_takes_the_rest():
    moe, pages, overlap = _plan(kv_tokens=32)
    assert pages == 4  # 32 tokens, page 10 -> 4 pages (40 tokens)
    assert moe == (2_000 - 4 * 10) // 100  # 19
    assert overlap is True
    assert moe * 100 + pages * 10 <= 2_000


def test_growing_kv_shrinks_moe_and_does_not_exceed_the_budget():
    small_moe, small_pages, _ = _plan(kv_tokens=32)
    big_moe, big_pages, _ = _plan(kv_tokens=1_000)
    assert big_pages > small_pages
    assert big_moe < small_moe
    assert big_moe >= 8
    assert big_moe * 100 + big_pages * 10 <= 2_000


def test_full_expert_cache_does_not_inflate_the_kv_pool():
    # Experts hit the cap; leftover bytes stay free instead of becoming extra KV pages.
    moe, pages, _ = _plan(kv_tokens=20, max_slots=8, total_experts=8, moe_floor=8)
    assert moe == 8
    assert pages == 2


def test_kv_pin_below_the_moe_floor_raises():
    with pytest.raises(ValueError, match="below the floor"):
        _plan(kv_tokens=1_900, moe_floor=8)


def test_max_tokens_keeps_the_moe_floor():
    fit = max_elastic_tokens(
        budget_bytes=2_000,
        per_expert_bytes=100,
        cache_per_page=10,
        page_size=10,
        moe_floor=8,
    )
    # 2000 - 8*100 = 1200 bytes -> 120 pages -> 1200 tokens.
    assert fit == 1_200
    moe, pages, _ = _plan(kv_tokens=fit)
    assert pages * 10 == fit
    assert moe == 8


def test_elastic_state_replans_the_same_budget():
    ek = ElasticKv(
        budget_bytes=2_000,
        per_expert_bytes=100,
        cache_per_page=10,
        page_size=10,
        num_experts=4,
        total_experts=50,
        max_slots=50,
        prefill_overlap=True,
        moe_floor=8,
        initial_tokens=40,
        max_tokens=1_200,
    )
    moe, pages, _ = ek.plan(40)
    grown, grown_pages, _ = ek.plan(ek.max_tokens)
    assert pages * ek.page_size == 40
    assert grown < moe
    assert grown_pages * ek.page_size == ek.max_tokens


def test_grow_pays_for_kv_pages_only():
    pe = 1024 * 1024
    ek = ElasticKv(
        budget_bytes=2000 * pe,
        per_expert_bytes=pe,
        cache_per_page=pe,
        page_size=64,
        num_experts=4,
        total_experts=5_000,
        max_slots=5_000,
        prefill_overlap=True,
        moe_floor=8,
        initial_tokens=32_768,
        max_tokens=262_144,
    )
    short, _, _ = ek.plan(32_768, activation_tokens=32_768)
    short_kv, _, _ = ek.plan(32_768)
    long, _, _ = ek.plan(65_536, activation_tokens=65_536)
    long_kv, _, _ = ek.plan(65_536)
    # Fixed scratch, not a per-token residual: 64k gives up the same extra slots as 32k.
    assert short_kv - short == 384
    assert long_kv - long == 384
    assert long < short


def test_grow_covers_several_requests_until_the_ceiling():
    # 40k + 40k fits under 100k; the pool grows once to their sum.
    assert elastic_grow_tokens(32_000, [40_000, 40_000], 100_000) == 80_000
    # The second 200k does not fit beside the first under 256k, so the pool grows to the first.
    assert elastic_grow_tokens(32_000, [200_000, 200_000], 262_144) == 200_000
    # Already covered: no rebuild.
    assert elastic_grow_tokens(80_000, [40_000, 40_000], 100_000) is None
    # A request above the ceiling is ignored; the one that fits still grows the pool.
    assert elastic_grow_tokens(32_000, [300_000, 40_000], 100_000) == 40_000
    assert elastic_grow_tokens(32_000, [], 100_000) is None


def test_elastic_kv_shrink_target_restores_initial_geometry():
    ek = ElasticKv(
        budget_bytes=2000 * 1024,
        per_expert_bytes=1024,
        cache_per_page=1024,
        page_size=64,
        num_experts=4,
        total_experts=5_000,
        max_slots=5_000,
        prefill_overlap=True,
        moe_floor=8,
        initial_tokens=32_768,
        max_tokens=262_144,
        initial_moe_slots=15_360,
    )
    assert ek.shrink_target(32_768) is None
    assert ek.shrink_target(16_384) is None
    assert ek.shrink_target(201_152) == (15_360, 512)

    ek_auto = ElasticKv(
        budget_bytes=2000 * 1024,
        per_expert_bytes=1024,
        cache_per_page=1024,
        page_size=64,
        num_experts=4,
        total_experts=5_000,
        max_slots=5_000,
        prefill_overlap=True,
        moe_floor=8,
        initial_tokens=32_768,
        max_tokens=262_144,
    )
    moe, pages = ek_auto.shrink_target(65_536)
    expected_moe, expected_pages, _ = ek_auto.plan(32_768)
    assert (moe, pages) == (expected_moe, expected_pages)


def test_scheduler_idle_shrink_timeout_and_trigger(monkeypatch):
    from types import SimpleNamespace
    from freetoken.env import ENV
    from freetoken.scheduler.scheduler import Scheduler

    ek = ElasticKv(
        budget_bytes=2000 * 1024,
        per_expert_bytes=1024,
        cache_per_page=1024,
        page_size=64,
        num_experts=4,
        total_experts=5_000,
        max_slots=5_000,
        prefill_overlap=True,
        moe_floor=8,
        initial_tokens=32_768,
        max_tokens=262_144,
        initial_moe_slots=15_360,
    )
    rebuilt = []
    replied = []
    sched = object.__new__(Scheduler)
    sched.config = SimpleNamespace(offline_mode=False)
    sched.engine = SimpleNamespace(elastic_kv=ek, num_pages=512)
    sched._kv_wait = []
    sched.prefill_manager = SimpleNamespace(runnable=False)
    sched.decode_manager = SimpleNamespace(runnable=False)
    sched.rebuild_cache = lambda **kw: rebuilt.append(kw)
    sched._log_cache_geometry = lambda _event: None
    sched._reply_rebuild = lambda rid, status: replied.append((rid, status))

    monkeypatch.setattr(ENV.ELASTIC_KV_SHRINK_IDLE_SEC, "value", 60.0)
    # At initial size: no timeout, no shrink
    assert sched.idle_timeout_ms() is None
    sched._maybe_shrink_kv()
    assert rebuilt == []

    # Expanded to 201K: timeout active and _maybe_shrink_kv restores initial geometry
    sched.engine.num_pages = 3143
    assert sched.idle_timeout_ms() == 60_000
    sched._maybe_shrink_kv()
    assert rebuilt == [{"moe_cache_size": 15_360, "num_pages": 512}]
    assert replied == [("elastic_kv", "ok")]

    # Disabled via env=0
    monkeypatch.setattr(ENV.ELASTIC_KV_SHRINK_IDLE_SEC, "value", 0.0)
    assert sched.idle_timeout_ms() is None
