"""The GDN host bank's free-list is a plain list, so a slot returned twice is a slot
serving two boundaries at once. These tests pin the guard and the chunk-chain case
that can reach it."""

import torch

from freetoken.core import Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.scheduler.cache import CacheManager


class _Pool:
    """The real ``_return_slots`` guard over plain lists, without allocating a bank."""

    # Device side, for CacheManager.check_integrity's GDN-slot bound: the tree holds no
    # device snapshots in these fixtures, so all usable device slots read free.
    num_slots = 32

    @property
    def num_free_slots(self):
        return len(self._free_slots)

    _num_host_slots = 32

    @property
    def num_host_slots(self):
        return self._num_host_slots

    @property
    def num_free_host_slots(self):
        return len(self._free_host_slots)

    def __init__(self, slots=8, held=(), held_state=()):
        self._free_slots = list(range(1, slots))
        self._free_host_slots = list(range(1, slots))
        # Slots a live request owns are absent from the free-list, or every release below
        # would (correctly) look like a double free.
        for s in held:
            self._free_host_slots.remove(s)
        for s in held_state:
            self._free_slots.remove(s)
        self.double_frees = 0

    def free(self, slots):
        return LinearStatePool._return_slots(self, slots, self._free_slots, "state")

    def free_host(self, slots):
        return LinearStatePool._return_slots(self, slots, self._free_host_slots, "host bank")


class _Manager:
    def __init__(self):
        self.linear_state_pool = _Pool()


def _req(uid, host_tracks=None):
    req = Req(
        input_ids=torch.zeros(4, dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=8,
        uid=uid,
        sampling_params=SamplingParams(),
        cache_handle=None,
    )
    req.mamba_ping_pong = None
    req.linear_slot_idx = None
    req.mamba_host_tracks = host_tracks
    return req


def test_a_slot_returned_twice_is_listed_once():
    pool = _Pool()
    pool._free_host_slots = [5, 6]
    pool.free_host([5, 6, 7])
    assert sorted(pool._free_host_slots) == [5, 6, 7]
    assert pool.double_frees == 2, "the aliasing must be counted, not swallowed"


def test_a_clean_release_costs_nothing():
    pool = _Pool()
    before = list(pool._free_host_slots)
    pool.free_host([9, 10])
    assert pool._free_host_slots == before + [9, 10]
    assert pool.double_frees == 0


def test_tensor_and_int_forms_still_work():
    pool = _Pool()
    pool.free_host(torch.tensor([9, 10]))
    pool.free_host(11)
    assert {9, 10, 11} <= set(pool._free_host_slots)


def test_a_shared_chunk_chain_releases_its_boundaries_once():
    """Overlap scheduling builds the next chunk's Req from this chunk's track list *by
    identity*, so two live Reqs can hold the same boundaries. Releasing both must not
    put a slot on the free-list twice -- that is what hands one checkpoint to two
    requests."""
    manager = _Manager.__new__(_Manager)
    manager.linear_state_pool = _Pool(slots=32, held=[3, 4])
    shared = [(8064, 3), (12160, 4)]
    chunk, continuation = _req(1, shared), _req(2, shared)

    CacheManager._free_req_slots(manager, chunk)
    CacheManager._free_req_slots(manager, continuation)

    assert sorted(manager.linear_state_pool._free_host_slots) == list(range(1, 32))
    assert manager.linear_state_pool.double_frees == 0


def test_state_slots_are_still_idempotent_through_the_same_path():
    manager = _Manager.__new__(_Manager)
    manager.linear_state_pool = _Pool(slots=8, held_state=[2, 3, 4])
    req = _req(3)
    req.mamba_ping_pong = [2, 3]
    req.linear_slot_idx = 4
    CacheManager._free_req_slots(manager, req)
    CacheManager._free_req_slots(manager, req)
    assert manager.linear_state_pool.double_frees == 0
    assert sorted(manager.linear_state_pool._free_slots) == [1, 2, 3, 4, 5, 6, 7]


class _Tree:
    """The prefix-cache surface ``CacheManager.check_integrity`` reads."""

    full_evictable = 0
    full_protected = 0
    mamba_evictable_size = 0
    mamba_protected = 0

    def __init__(self, host_nodes):
        self.mamba_host_evictable = host_nodes
        self.mamba_host_protected = 0

    def check_integrity(self):
        return None


class _Probe:
    """A hybrid cache manager over a bank of ``capacity`` slots, with ``free`` listed as
    free and ``tree_host`` boundaries owned by the tree. Only the arithmetic the real
    check cares about is modelled; the check itself is production code."""

    is_hybrid = True
    is_swa = False
    page_size = 64
    capacity = 31  # slot 0 is the reserved padding sink
    # No KV pages in these fixtures, so the shared page tail of the check is satisfied.
    num_pages = 0
    free_slots = torch.zeros(0, dtype=torch.long)

    def __init__(self, free, tree_host):
        pool = _Pool(slots=32)
        pool._free_host_slots = list(free)
        pool._num_host_slots = self.capacity + 1
        self.linear_state_pool = pool
        self.prefix_cache = _Tree(tree_host)

    check_integrity = CacheManager.check_integrity


def test_a_stale_release_of_a_tree_owned_boundary_would_wedge_the_engine():
    """The hang, in one test.

    ``Scheduler.run_when_idle`` calls ``cache_manager.check_integrity()`` every time the
    scheduler has nothing to do, and ``launch._run_scheduler`` catches only
    ``KeyboardInterrupt`` -- so an AssertionError here kills that rank while its three
    peers sit in a gloo/pynccl collective configured with ``--distributed-timeout
    2592000`` (30 days, and ``use_pynccl`` defaults to True). Nothing errors; the
    frontend's ``wait_for_ack`` has no timeout either, so every request hangs with no
    response until the process is restarted.

    The reachable sequence: the tree owns a boundary, one holder releases it, and a
    sibling of the chunked-prefill chain releases the same slot again.
    """
    manager = _Probe(free=[7], tree_host=30)  # consistent: 1 free + 30 tree-owned == 31
    manager.check_integrity()  # the idle check passes before the corruption

    manager.linear_state_pool._free_host_slots.append(7)  # what the unguarded free_host did
    import pytest

    with pytest.raises(AssertionError, match="mamba bank leak"):
        manager.check_integrity()


def test_the_guard_lets_the_scheduler_survive_that_same_release():
    import pytest

    manager = _Probe(free=[7], tree_host=30)
    manager.linear_state_pool.free_host(7)  # the stale sibling's release, via the guard
    assert manager.linear_state_pool._free_host_slots == [7]
    assert manager.linear_state_pool.double_frees == 1
    manager.check_integrity()  # no AssertionError: the rank, and the engine, stay up

    # A legitimate release is still honoured, guard intact.
    manager.linear_state_pool.free_host(8)
    assert manager.linear_state_pool._free_host_slots == [7, 8]
