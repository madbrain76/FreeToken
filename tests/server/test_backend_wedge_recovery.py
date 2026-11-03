"""A dead or wedged backend must produce an error, not silence.

The rank-loss wedge outlives every other bug here: the surviving ranks sit in a
collective allowed to run for ``--distributed-timeout`` (30 days as launched, and
``use_pynccl`` puts that on the gloo group), the frontend's ``wait_for_ack`` had no
timeout at all, and the post-death SIGTERM parks the process in uvicorn's connection
drain -- which waits on precisely those requests. Clients hang for tens of minutes and
only a restart helps. These drive the real frontend machinery.
"""

import asyncio

from freetoken.server import api_server as AS


def _state():
    return AS.FrontendManager(config=None, send_tokenizer=None, recv_tokenizer=None)


def _admit(state, uid):
    """What new_user() does for the ack plumbing, minus stats and the admission gate."""
    state.ack_map[uid] = []
    state.event_map[uid] = asyncio.Event()


async def _drain(state, uid):
    return [ack async for ack in state.wait_for_ack(uid)]


def test_wait_for_ack_fails_a_request_the_engine_never_answers(monkeypatch):
    monkeypatch.setattr(AS.ENV.ACK_IDLE_TIMEOUT, "value", 0.05)
    state = _state()
    _admit(state, 1)

    # wait_for, not a bare run(): if the breaker ever regresses, this test must FAIL rather
    # than hang the suite for as long as the client is willing to wait.
    replies = asyncio.run(asyncio.wait_for(_drain(state, 1), timeout=2.0))

    assert len(replies) == 1
    assert replies[0].finished and replies[0].error_code == "backend_silent"
    assert "no output for" in replies[0].error
    # Bookkeeping is cleaned up exactly as it is for a normal terminal reply.
    assert 1 not in state.ack_map and 1 not in state.event_map


def test_acks_that_keep_arriving_are_never_cut_off(monkeypatch):
    monkeypatch.setattr(AS.ENV.ACK_IDLE_TIMEOUT, "value", 5.0)
    state = _state()
    _admit(state, 2)

    async def scenario():
        task = asyncio.create_task(_drain(state, 2))
        for text in ("a", "b"):
            await asyncio.sleep(0.01)
            state.ack_map[2].append(
                AS.UserReply(uid=2, incremental_output=text, finished=False)
            )
            state.event_map[2].set()
        await asyncio.sleep(0.01)
        state.ack_map[2].append(AS.UserReply(uid=2, incremental_output="c", finished=True))
        state.event_map[2].set()
        return await asyncio.wait_for(task, timeout=1.0)

    replies = asyncio.run(scenario())
    assert [r.incremental_output for r in replies] == ["a", "b", "c"]
    assert not any(r.error for r in replies)


def test_idle_timeout_of_zero_restores_the_wait_forever_behaviour(monkeypatch):
    monkeypatch.setattr(AS.ENV.ACK_IDLE_TIMEOUT, "value", 0.0)
    state = _state()
    _admit(state, 3)

    async def scenario():
        task = asyncio.create_task(_drain(state, 3))
        await asyncio.sleep(0.15)
        assert not task.done(), "0 must disable the breaker"
        state.ack_map[3].append(AS.UserReply(uid=3, incremental_output="late", finished=True))
        state.event_map[3].set()
        return await asyncio.wait_for(task, timeout=1.0)

    assert [r.incremental_output for r in asyncio.run(scenario())] == ["late"]


def test_a_dead_backend_wakes_every_parked_request():
    state = _state()
    _admit(state, 4)
    _admit(state, 5)

    async def scenario():
        state._loop = asyncio.get_running_loop()
        tasks = [asyncio.create_task(_drain(state, uid)) for uid in (4, 5)]
        await asyncio.sleep(0)  # let both consumers park on their events
        state.fail_admitted_requests("FreeToken backend died: boom")
        return await asyncio.wait_for(asyncio.gather(*tasks), timeout=1.0)

    for replies in asyncio.run(scenario()):
        assert replies[-1].finished and replies[-1].error_code == "backend_gone"
        assert "backend died" in replies[-1].error
    assert not state.ack_map and not state.event_map


def test_failing_requests_before_the_listener_starts_is_harmless():
    state = _state()
    _admit(state, 6)
    state.fail_admitted_requests("boom")  # no loop yet: must not raise
    assert state.ack_map[6] == []


def test_the_shutdown_still_escalates_when_the_drain_never_finishes(monkeypatch):
    """The bound on uvicorn's graceful drain.

    ``_SHUTTING_DOWN`` is set by the very SIGTERM we send, so the escalation is gated on
    having sent it -- otherwise the one case it exists for would be skipped.
    """
    killed, term = [], []
    monkeypatch.setattr(AS.os, "kill", lambda pid, sig: killed.append(sig))
    monkeypatch.setattr(AS.signal, "SIGTERM", "TERM", raising=False)
    monkeypatch.setattr(AS.signal, "SIGKILL", "KILL", raising=False)
    monkeypatch.setattr(AS, "_reap_backend_workers", lambda processes, timeout=5.0: None)

    timer = AS._exit_after_backend_death(0.01, force_grace_s=0.02)
    deadline = asyncio.run(asyncio.sleep(0))  # keep import-time side effects obvious
    import time

    for _ in range(300):
        time.sleep(0.01)
        if killed == ["TERM", "KILL"]:
            break
    timer.cancel()
    assert killed == ["TERM", "KILL"], f"expected SIGTERM then SIGKILL, got {killed}"
