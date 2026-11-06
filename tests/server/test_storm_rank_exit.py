"""The rank that gives up has to actually leave.

On 2026-11-06 all four ranks logged "storm breaker gave up" and the serve stayed up: the handler
raised ``SystemExit``, which hands control to interpreter shutdown -- and shutdown joins the
rank's threads and runs the CUDA/NCCL destructors while its collectives are still in flight. One
rank died after 13s, the rest never did, so the launcher's restart loop waited forever on a child
that had already announced it is broken. The property under test is the process's exit, so these
run the real helper in a real child process.
"""

import os
import subprocess
import sys
import textwrap

import pytest

from freetoken.storm import STORM_EXIT_CODE


def _run(body: str) -> subprocess.CompletedProcess:
    script = textwrap.dedent(
        """
        import threading
        # What interpreter shutdown joins, and what a rank mid-collective never returns from.
        threading.Thread(target=threading.Event().wait, name="collective-in-flight").start()
        """
    ) + textwrap.dedent(body)
    try:
        return subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            "the rank is still alive 30s after giving up: something is waiting on interpreter "
            f"shutdown again (the original bug). stdout={exc.stdout!r} stderr={exc.stderr!r}"
        )


def _output(done: subprocess.CompletedProcess) -> str:
    """The engine's console handler writes to stdout; a crash trace goes to stderr."""
    return (done.stdout or "") + (done.stderr or "")


def test_the_rank_exits_with_the_storm_code_despite_a_parked_thread():
    done = _run(
        """
        from freetoken.server import launch
        launch._storm_hard_exit(2, RuntimeError("8 distinct requests tripped a breaker"))
        print("REACHED-AFTER-EXIT")
        """
    )

    assert done.returncode == STORM_EXIT_CODE, f"expected {STORM_EXIT_CODE}: {_output(done)[-2000:]}"
    assert "storm breaker gave up" in _output(done), "the give-up must still be on the record"
    assert "REACHED-AFTER-EXIT" not in _output(done), "os._exit must not fall through"


def test_a_dead_log_pipe_cannot_keep_a_given_up_rank_alive():
    """The transcript is a pipe to tee; a writer that stops reading must not own our lifetime."""
    done = _run(
        """
        import logging
        from freetoken.server import launch

        class Dead(logging.Handler):
            def emit(self, record): raise OSError("log pipe is full")
            def flush(self): raise OSError("log pipe is full")

        logging.getLogger("freetoken.server.launch").handlers = [Dead()]
        launch._storm_hard_exit(3, RuntimeError("boom"))
        print("REACHED-AFTER-EXIT")
        """
    )

    assert done.returncode == STORM_EXIT_CODE, f"expected {STORM_EXIT_CODE}: {_output(done)[-2000:]}"
    assert "REACHED-AFTER-EXIT" not in _output(done)
