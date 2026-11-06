from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
import os
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from freetoken.distributed import DistributedInfo
from freetoken.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs
    from .supervisor import BackendHandle


def _report_startup_error(ack_queue: mp.Queue, exc: BaseException) -> None:
    """Tell the parent WHY this worker is dying — push an ("error", reason) ack before it exits,
    so the supervisor reports the real cause (e.g. a config ValueError) instead of the generic
    "backend worker … exited during load". Best-effort and flushed (close + join_thread), since
    the process is about to terminate; a failure to report must never mask the original error."""
    try:
        ack_queue.put(("error", f"{type(exc).__name__}: {exc}"))
        ack_queue.close()
        ack_queue.join_thread()
    except Exception:  # noqa: BLE001 -- reporting is a nicety; never shadow the real exception
        pass


def _detach_process_group() -> None:
    """Shell mode only: move this worker out of the terminal's foreground process group.

    The shell binds ^C to "cancel this turn", but a terminal delivers SIGINT to the whole
    foreground group — which, with the engine running in this same process, includes the
    workers. They would take the same ^C and exit (``_run_scheduler`` below stops gracefully on
    KeyboardInterrupt), leaving the shell chatting with a dead engine. Nothing depends on the
    signal reaching them: the parent tears the workers down explicitly on every stop path
    (uvicorn's lifespan, plus the SIGTERM/SIGHUP handler and the reap backstop in api_server).

    ``ft serve`` keeps the default — uvicorn owns ^C there, and the group-wide delivery is part
    of how it stops."""
    try:
        os.setpgrp()
    except OSError:  # no job control (already a group leader / unusual environment)
        pass


def _run_tokenize_worker(detach: bool, **kwargs) -> None:
    """Module-level so it survives the spawn pickle; exists only to detach the group first."""
    if detach:
        _detach_process_group()
    from freetoken.tokenizer import tokenize_worker

    tokenize_worker(**kwargs)


def _run_vision_encoder(detach: bool, ack_queue: mp.Queue, model_path: str, mm, dtype, **addrs) -> None:
    """The --mm-encoder-weights cpu process; started with no CUDA device visible (see _cuda_hidden)."""
    if detach:
        _detach_process_group()
    from freetoken.mm.cpu_encoder import CpuVisionEncoder, configure_encoder_process, serve_vision_encoder

    try:
        threads = configure_encoder_process(mm)
        encoder = CpuVisionEncoder.from_checkpoint(model_path, mm, dtype)
    except Exception as exc:  # noqa: BLE001 -- surface the reason, then let it propagate
        _report_startup_error(ack_queue, exc)
        raise
    init_logger(__name__, "vision_cpu").info(
        f"CPU vision encoder ready: {threads} threads, image cache {mm.encoder_cache_mb} MiB"
    )
    serve_vision_encoder(encoder, ack_queue=ack_queue, **addrs)


@contextlib.contextmanager
def _cuda_hidden():
    """A spawned child copies this environment at start: with no device visible the CPU encoder can never
    open a CUDA context, which would take hundreds of MiB from GPU 0's pools."""
    old = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old


def _storm_hard_exit(rank: int, exc: BaseException) -> None:
    """Put this rank down without running any of its teardown.

    ``SystemExit`` is not enough. It hands control to interpreter shutdown, which joins threads
    and runs atexit -- for a rank that means the CUDA and NCCL destructors while its collectives
    are still in flight. Observed on 4x RTX 5060 Ti: all four ranks logged the give-up, one took
    13s to die and the rest never did, so the serve kept holding the port and the frame buffer
    and the launcher's restart loop waited forever on a child that had already told us it is
    broken. Discarding the process *is* the remedy, so there is nothing worth cleaning up: flush
    what the post-mortem needs and let the kernel do the rest.
    """
    from freetoken.storm import StormUnrecoverable

    logger = init_logger(__name__)
    logger.error(f"rank{rank}: storm breaker gave up: {exc}")
    # The transcript is a pipe to tee, so an unflushed record dies with os._exit().
    for handler in list(getattr(logger, "handlers", ())):
        try:
            handler.flush()
        except Exception:  # noqa: BLE001 -- a dead log pipe must not keep the rank alive
            pass
    os._exit(StormUnrecoverable.exit_code)


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    if args.shell_mode:
        _detach_process_group()

    # published (not bound) here: the engine binds it after the allocator setup
    from freetoken.gpu_select import set_assigned_gpu

    # resolved UUIDs when we have them, the raw --gpu entries when NVML could not resolve them, else one CUDA ordinal per rank
    targets = args.gpu_assigned or args.gpu or tuple(str(r) for r in range(args.tp_info.size))
    set_assigned_gpu(targets[args.tp_info.rank])

    import torch
    from freetoken.scheduler import Scheduler
    from freetoken.storm import StormUnrecoverable

    if args.tp_info.is_primary():
        from freetoken.utils.progress import set_progress_sink

        set_progress_sink(
            lambda desc, done, total: ack_queue.put(("progress", desc, done, total))
        )

    with torch.inference_mode():
        try:
            scheduler = Scheduler(args)
            scheduler.sync_all_ranks()
        except Exception as exc:  # noqa: BLE001 -- surface the reason, then let it propagate
            # A startup failure (bad config, OOM, corrupt weights) would otherwise reach the
            # parent only as a dead process -> a generic "exited during load". Push the real
            # reason first so the supervisor (and the desktop failure modal) can surface it;
            # the traceback still prints and the process still exits non-zero.
            _report_startup_error(ack_queue, exc)
            raise

        if args.tp_info.is_primary():
            # Report the real per-unit cache VRAM costs (KV/expert/mamba), the device-wide free
            # VRAM, and the per-pool rebuild floors before the ready ack, so the supervisor has
            # them (and the desktop's slider bounds) by the time the gate flips. Optional +
            # best-effort: a failure here must never keep the model from serving, and older
            # consumers ignore ("meta", …).
            try:
                from freetoken.kvcache.cache_status import compute_cache_status_meta

                meta = compute_cache_status_meta(scheduler.engine)
                # the parent must not touch CUDA to learn this
                meta["gpus"] = scheduler.gpus
                ack_queue.put(("meta", meta))
            except Exception:  # noqa: BLE001 -- metadata is a nicety; readiness is not
                pass
            ack_queue.put("Scheduler is ready")
            # The supervisor stops draining ack_queue once ready, so uninstall the sink:
            # runtime cache rebuilds re-run the graph capture (which emits progress) and
            # would otherwise push onto a queue nobody reads for the server's lifetime.
            set_progress_sink(None)

        if args.silent_output:
            logging.disable(logging.INFO)

        try:
            scheduler.run_forever()
        except KeyboardInterrupt:
            logger = init_logger(__name__)
            if args.tp_info.is_primary():
                print()  # for a clean newline after ^C
                logger.info("Scheduler exiting gracefully...")
            scheduler.shutdown()
        except StormUnrecoverable as exc:
            # Exit with a code the API server recognises, so it takes the serve down at once
            # instead of parking in uvicorn's connection drain waiting on dead backends. The
            # exit is unconditional: see _storm_hard_exit for why raising was not enough.
            _storm_hard_exit(args.tp_info.rank, exc)


def launch_server(
    run_shell: bool = False,
    argv: list[str] | None = None,
    prog: str | None = None,
) -> None:
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(
        sys.argv[1:] if argv is None else argv,
        run_shell,
        prog=prog,
    )
    logger = init_logger(__name__, "initializer")

    if server_args.gpu:
        # resolve here so a typo is one clear error before any worker spawns
        from freetoken.gpu_select import resolve_gpu_uuids

        try:
            server_args = replace(server_args, gpu_assigned=resolve_gpu_uuids(server_args.gpu))
        except ValueError as exc:
            raise SystemExit(f"{prog or 'ft serve'}: error: {exc}") from exc
        logger.info(
            f"--gpu {','.join(server_args.gpu)} -> "
            f"{', '.join(server_args.gpu_assigned) if server_args.gpu_assigned else 'resolved at CUDA init (no NVML)'}"
        )

    def start_subprocess() -> "BackendHandle":
        import multiprocessing as mp

        from .supervisor import BackendHandle

        mp.set_start_method("spawn", force=True)
        detach = server_args.shell_mode  # see _detach_process_group

        world_size = server_args.tp_info.size
        ack_queue: mp.Queue = mp.Queue()
        processes: list[mp.Process] = []

        for i in range(world_size):
            new_args = replace(server_args, tp_info=DistributedInfo(i, world_size))
            p = mp.Process(
                target=_run_scheduler,
                args=(new_args, ack_queue),
                daemon=False,
                name=f"freetoken-TP{i}-scheduler",
            )
            p.start()
            processes.append(p)

        encoder_addr = None
        if server_args.mm.encoder_out_of_process and "image" in server_args.served_modalities:
            encoder_addr = server_args.zmq_encoder_addr
            p = mp.Process(
                target=_run_vision_encoder,
                kwargs={
                    "detach": detach,
                    "ack_queue": ack_queue,
                    "model_path": server_args.model_path,
                    "mm": server_args.mm,
                    "dtype": server_args.dtype,
                    "addr": encoder_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                },
                daemon=False,
                name="freetoken-vision-cpu",
            )
            with _cuda_hidden():
                p.start()
            processes.append(p)

        num_tokenizers = server_args.num_tokenizer
        p = mp.Process(
            target=_run_tokenize_worker,
            kwargs={
                "detach": detach,
                "tokenizer_path": server_args.model_path,
                "addr": server_args.zmq_detokenizer_addr,
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,
                "local_bs": 1,
                "mm": server_args.mm,
                "create": server_args.tokenizer_create_addr,
                "tokenizer_id": num_tokenizers,
                "ack_queue": ack_queue,
                "encoder_addr": encoder_addr,
            },
            daemon=False,
            name="freetoken-detokenizer-0",
        )
        p.start()
        processes.append(p)
        for i in range(num_tokenizers):
            p = mp.Process(
                target=_run_tokenize_worker,
                kwargs={
                    "detach": detach,
                    "tokenizer_path": server_args.model_path,
                    "addr": server_args.zmq_tokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "mm": server_args.mm,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": i,
                    "ack_queue": ack_queue,
                    "encoder_addr": encoder_addr,
                },
                daemon=False,
                name=f"freetoken-tokenizer-{i}",
            )
            p.start()
            processes.append(p)

        # Expected ready acks: 1 primary scheduler + num_tokenizers + 1 detokenizer (+ the CPU vision encoder).
        return BackendHandle(
            ack_queue=ack_queue,
            processes=processes,
            expected_acks=num_tokenizers + 2 + (encoder_addr is not None),
        )

    run_api_server(server_args, start_subprocess, run_shell=run_shell)


if __name__ == "__main__":
    launch_server()
