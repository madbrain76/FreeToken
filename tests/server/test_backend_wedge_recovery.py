"""A dead or wedged backend must produce an error, not silence.

The rank-loss wedge outlives every other bug here: the surviving ranks sit in a
collective allowed to run for ``--distributed-timeout`` (30 days as launched, and
``use_pynccl`` puts that on the gloo group), the frontend's ``wait_for_ack`` had no
timeout at all, and the post-death SIGTERM parks the process in uvicorn's connection
drain -- which waits on precisely those requests. Clients hang for tens of minutes and
only a restart helps. These drive the real frontend machinery.
"""

import asyncio
import signal
import os
import time
from types import SimpleNamespace

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


def _wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_a_storm_give_up_exits_without_waiting_for_connected_clients(monkeypatch):
    """The engine said it is broken; every live connection is one it can never answer.

    The ordinary post-death path SIGTERMs and lets uvicorn drain, bounded only by
    ``BACKEND_DEATH_EXIT_GRACE_S`` plus ``BACKEND_DEATH_FORCE_GRACE_S``. For a deliberate
    give-up that wait is pure latency before the restart the operator has to do anyway, so the
    exit must be bounded by a constant and must not depend on the clients at all.
    """
    signals, exits, killed_pids = [], [], []

    monkeypatch.setattr(
        AS,
        "_GLOBAL_STATE",
        SimpleNamespace(backend_processes=[SimpleNamespace(pid=4242), SimpleNamespace(pid=None), object()]),
    )
    monkeypatch.setattr(AS, "STORM_EXIT_FLUSH_S", 0.01)
    monkeypatch.setattr(AS.os, "kill", lambda pid, sig: signals.append(sig))
    monkeypatch.setattr(AS.os, "_exit", lambda code: exits.append(code))
    monkeypatch.setattr(AS, "_kill_backend_workers_detached", lambda pids: killed_pids.append(pids))
    monkeypatch.setattr(AS.signal, "SIGTERM", "TERM", raising=False)
    monkeypatch.setattr(AS.signal, "SIGKILL", "KILL", raising=False)
    AS._SHUTTING_DOWN.clear()

    started = time.monotonic()
    try:
        timers = AS._exit_after_backend_death(
            AS.BACKEND_DEATH_EXIT_GRACE_S, force_grace_s=AS.BACKEND_DEATH_FORCE_GRACE_S,
            immediate=True,
        )
        reached = _wait_for(lambda: bool(exits))
    finally:
        timers.cancel()
    elapsed = time.monotonic() - started

    assert reached, f"the give-up never exited: signals={signals}"
    assert exits == [AS.STORM_EXIT_CODE], "the launcher relies on the storm exit code"
    assert signals == [], f"must not SIGTERM, that is what parks us in the drain: {signals}"
    # The ranks must be put down (they would otherwise hold the GPUs for a whole month of
    # distributed timeout) and only the live pids are worth signalling.
    assert killed_pids == [[4242]], f"the ranks it leaves behind must not keep the GPUs: {killed_pids}"
    assert elapsed < AS.BACKEND_DEATH_EXIT_GRACE_S, (
        f"waited {elapsed:.2f}s, as long as the drain path it replaces"
    )


def test_a_give_up_that_never_finishes_still_takes_the_process_down(monkeypatch):
    """The 2026-11-06 failure mode: it logged the exit, hung, and the serve lived on.

    All four ranks announced the give-up; one died after 13s and the rest never did, so the
    frontend printed "exiting now without waiting for connected clients" and then sat holding the
    port -- and the launcher's restart loop, waiting on a child that never exits, never restarted
    anything. Whatever is left between the announcement and the exit must therefore be unable to
    hold the process: the path arms a kernel-level backstop of its own before it touches them.
    """
    signals, exits = [], []

    def hang_forever(pids):
        time.sleep(30)  # what a rank wedged in driver teardown does to a caller that waits

    monkeypatch.setattr(
        AS, "_GLOBAL_STATE", SimpleNamespace(backend_processes=[SimpleNamespace(pid=4242)])
    )
    monkeypatch.setattr(AS, "STORM_EXIT_FLUSH_S", 0.01)
    monkeypatch.setattr(AS, "STORM_EXIT_SELF_KILL_S", 0.2)
    monkeypatch.setattr(AS, "_kill_backend_workers_detached", hang_forever)
    monkeypatch.setattr(AS.os, "_exit", lambda code: exits.append(code))
    monkeypatch.setattr(AS.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    AS._SHUTTING_DOWN.clear()

    self_kill = (os.getpid(), signal.SIGKILL)
    started = time.monotonic()
    try:
        timers = AS._exit_after_backend_death(AS.BACKEND_DEATH_EXIT_GRACE_S, immediate=True)
        backstopped = _wait_for(lambda: self_kill in signals and not exits, timeout=3.0)
    finally:
        timers.cancel()
    elapsed = time.monotonic() - started

    assert not exits, "the simulated wedge should have prevented the orderly exit"
    assert backstopped, f"nothing killed the wedged give-up: {signals}"
    assert all(sig != signal.SIGTERM for _, sig in signals), (
        "a SIGTERM is the drain this path exists to avoid"
    )
    assert elapsed < AS.BACKEND_DEATH_EXIT_GRACE_S, (
        f"the backstop took {elapsed:.2f}s, no better than the drain it replaces"
    )


def test_a_storm_give_up_stays_quiet_when_an_external_stop_got_there_first(monkeypatch):
    """An operator ^C owns the shutdown; the give-up must not hijack its drain."""
    exits = []
    monkeypatch.setattr(AS, "_GLOBAL_STATE", SimpleNamespace(backend_processes=[]))
    monkeypatch.setattr(AS, "STORM_EXIT_FLUSH_S", 0.0)
    monkeypatch.setattr(AS.os, "_exit", lambda code: exits.append(code))
    AS._SHUTTING_DOWN.set()
    try:
        timer = AS._exit_after_backend_death(10.0, immediate=True)
        assert not _wait_for(lambda: bool(exits), timeout=0.3)
        timer.cancel()
    finally:
        AS._SHUTTING_DOWN.clear()


def test_an_ordinary_backend_death_still_gets_its_graceful_drain(monkeypatch):
    """The no-wait path is for a known-broken engine only, not for every death."""
    signals = []
    monkeypatch.setattr(AS, "_GLOBAL_STATE", SimpleNamespace(backend_processes=[]))
    monkeypatch.setattr(AS.os, "kill", lambda pid, sig: signals.append(sig))
    monkeypatch.setattr(AS.signal, "SIGTERM", "TERM", raising=False)
    AS._SHUTTING_DOWN.clear()
    try:
        timer = AS._exit_after_backend_death(0.01, force_grace_s=0.0)
        assert _wait_for(lambda: bool(signals))
        assert signals == ["TERM"]
    finally:
        timer.cancel()
        AS._SHUTTING_DOWN.clear()
