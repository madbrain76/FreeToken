"""The GDN host bank's free-list is a plain list, so a slot returned twice is a slot
serving two boundaries at once. These tests pin the guard and the chunk-chain case
that can reach it."""

import torch

from freetoken.core import Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.scheduler.cache import CacheManager


class _Pool:
    """The real ``_return_slots`` guard over plain lists, without allocating a bank."""

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
