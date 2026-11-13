from __future__ import annotations

import os
import time

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from freetoken.core import Batch, Req
from freetoken.env import ENV
from freetoken.gpu_select import gpu_identity
from freetoken.storm import (
    DEGENERATE_REPEAT,
    NONFINITE_LOGITS,
    StormLedger,
    StormUnrecoverable,
    note_repeated_token,
    record_storm_event,
    row_fault,
)
from freetoken.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    CacheRebuildBackendMsg,
    CacheRebuildResultMsg,
    DetokenizeMsg,
    ErrorReplyMsg,
    ExitMsg,
    FlushCacheBackendMsg,
    FlushCacheResultMsg,
    PromptAdmittedMsg,
    UserMsg,
)
from freetoken.utils import (
    div_ceil,
    init_logger,
    load_eos_token_ids,
    load_tokenizer,
    load_toolcall_anchor_id,
)

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .interleave import DecodeInterleavePolicy
from .io import SchedulerIOMixin
from .mm import cut_image_spans, plan_mm_batch
from .prefill import ChunkedReq, PrefillManager, _LAYER_MAJOR_MAX_EXTEND_TOKENS
from .status import SchedulerStatusReporter
from .table import TableManager

if TYPE_CHECKING:
    from freetoken.engine import BatchSamplingArgs, ForwardOutput
    from freetoken.engine.spec_protocol import SpecRuntime


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


def _gib(n_bytes: int) -> str:
    return f"{n_bytes / (1 << 30):.2f} GiB"


def _supports_layer_major_prefill(config: SchedulerConfig) -> bool:
    model_config = config.model_config
    return bool(
        getattr(config, "prefill_layer_major", False)
        and model_config.model_type == "qwen4_exp"
        and config.tp_info.size in (1, 2, 4, 8)
        and config.moe_strategy == "offload"
        and getattr(model_config, "expert_quant", None) == "nvfp4"
        and getattr(config, "ple_backend", None) == "pinned"
    )


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D | None  # (req_mapping, seq_lens or -1); None on verify, whose drain writes the pending token


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from freetoken.engine import Engine

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)
        # sent on the readiness ack for /v1/stats gpus; a list so TP can add one entry per rank
        self.gpus = [gpu_identity(self.device.index)] if self.device.type == "cuda" else []

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.spec: SpecRuntime | None = self.engine.spec
        # ONE cache manager for every model (ShadowRadix layering): the shared page table is the
        # virtual full-token coordinate; model-specific tiers ride the plug-ins -- DSV4's
        # window/cmp/idx shadows via swa_pool, Gemma's swa via swa_pool, GDN state via
        # linear_state_pool. No model supplies its own manager.
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type,
            linear_state_pool=self.engine.linear_state_pool,
            swa_pool=self.engine.kv_cache,
            sliding_window_size=next(
                (g.sliding_window for g in config.model_config.kv_cache_group_specs() if g.is_swa),
                None,
            ) or getattr(self.engine.kv_cache, "sliding_window_size", None),
            radix_lookahead=self.spec is not None and self.spec.radix_lookahead,
        )
        token_headroom = config.decode_rows_per_req - 1
        # A donated snapshot is copied on the engine stream, where it is program-ordered against
        # the forwards rather than racing them from the scheduler stream.
        self.cache_manager.donate_stream = self.engine.stream
        # Breaker trips, to tell one unlucky request from an engine that cannot recover.
        self._storm = StormLedger()
        self.decode_manager = DecodeManager(config.page_size, token_headroom=token_headroom)
        # Bound how long a run of prefill steps may starve in-flight decodes. A long
        # prompt chunked at the window budget is hundreds of consecutive prefill steps;
        # measured on an 8xRTX4090 DSV4 deployment, that left an already-decoding
        # request unscheduled for 267-317 s. Unset reproduces the historical
        # prefill-first order exactly.
        self._interleave = DecodeInterleavePolicy(
            getattr(config, "decode_interleave_every", None)
        )
        self._bidirectional_mm = any(getattr(g, "bidirectional_mm_blocks", False) for g in config.model_config.attention_groups)
        self.prefill_manager = PrefillManager(
            self.cache_manager,
            self.table_manager,
            self.decode_manager,
            encoder_cache=self.engine.encoder_cache,
            keep_images_whole=self._bidirectional_mm,
            token_headroom=token_headroom,
        )
        self.layer_major = _supports_layer_major_prefill(config)
        if getattr(config, "prefill_layer_major", False) and not self.layer_major:
            logger.warning_rank0(
                "--prefill-layer-major requires Qwen3.8 text, TP in [1, 2, 4, 8], NVFP4 GPU offload, and the pinned PLE backend; using ordinary chunked prefill"
            )
        self.prefill_manager.layer_major = self.layer_major

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        # Abort acknowledgements are a terminal accounting barrier. Queue them while processing
        # inbound control messages, then flush only AFTER _process_last_data publishes any
        # sampled replies from the prior overlapped forward.
        self._pending_abort_acks: Set[int] = set()
        # With multiple tokenizer workers, an AbortBackendMsg and its earlier UserMsg can arrive
        # through different PUSH producers and be observed out of order. Preserve a bounded
        # tombstone so an abort-before-admission request can never be resurrected after its
        # terminal accounting acknowledgement has already been published.
        self._abort_tombstones: dict[int, None] = {}
        self._forward_iter = 0  # global forward counter; drives the SWA proactive-eviction cadence
        # The launched-but-not-yet-drained batch (overlap): set at the top of each overlap_loop
        # iteration so the abort handler can tell whether a request's forward is still in flight
        # (mark it, defer the free to _process_last_data) or not (free immediately). Stays None
        # in normal_loop, where a batch launches and drains within one iteration.
        self._last_data: ForwardData | None = None
        # A received-but-not-yet-executed runtime cache rebuild (CacheRebuildBackendMsg),
        # run at the next idle safe point in overlap_loop. None when no rebuild is pending.
        self._pending_rebuild: CacheRebuildBackendMsg | None = None
        # Requests whose prompt+generation does not fit the current KV pool. They are not in
        # prefill_manager, so a short request can still run; the pool grows once the scheduler
        # is idle. Every TP rank sees the same messages, so the list stays identical.
        self._kv_wait: list[UserMsg] = []
        # A /v1/cache/flush waiting for the idle safe point, with its time.monotonic() deadline.
        self._pending_flush: Tuple[FlushCacheBackendMsg, float] | None = None
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_ids = load_eos_token_ids(config.model_path, self.tokenizer)
        self.toolcall_anchor_id = None
        if config.special_token_ckpt and (
            self.cache_manager.is_hybrid or self.cache_manager.is_swa
        ):
            from freetoken.server.function_call_parser import toolcall_opener_for

            self.toolcall_anchor_id = load_toolcall_anchor_id(
                self.tokenizer,
                toolcall_opener_for(getattr(config, "tool_call_parser", "")),
            )
        self.token_pool = self.table_manager.token_pool
        # Floor the prefill chunk by the cache manager's cap (DSV4: ~half the window pool) so a
        # sliding-window cache chunks long prompts and frees out-of-window pages between chunks
        # instead of OOMing _alloc_window on a prompt longer than the window pool.
        _chunk_cap = self.cache_manager.prefill_chunk_budget
        self.prefill_budget = (
            min(config.max_extend_tokens, _chunk_cap) if _chunk_cap else config.max_extend_tokens
        )
        self.config = config
        self._model_is_mrope = config.model_config.model_is_mrope
        self._warned_cut_image = False
        moe_cache = getattr(self.engine, "moe_offload_cache", None)
        self.status_reporter = SchedulerStatusReporter(
            log=logger.info_rank0,
            decode_log_interval=config.decode_log_interval,
            moe_stats=self._moe_stats_line if config.moe_collect_stats and moe_cache is not None else None,
        )

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def _moe_stats_line(self) -> str | None:
        # on the engine stream, so the read and the zeroing fall between decode replays
        with torch.cuda.stream(self.engine.stream):
            return self.engine.moe_offload_cache.decode_stats_line()

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    @torch.inference_mode()
    def rebuild_cache(
        self,
        *,
        moe_cache_size: int | None = None,
        num_pages: int | None = None,
        num_mamba_slots: int | None = None,
        num_swa_pages: int | None = None,
    ) -> None:
        """Idle-only runtime cache rebuild: resize the MoE slot cache, KV pages, GDN (mamba) state
        pool, and/or the window pool (num_swa_pages), re-capture CUDA graphs, and re-thread the
        page managers (clearing the prefix cache on a KV/mamba/window resize). The caller MUST
        guarantee the scheduler is idle — no pending prefill, no running decode, no in-flight
        finished requests. All TP ranks must call this with identical arguments.
        """
        assert not self.prefill_manager.runnable, "rebuild requires no pending prefill"
        assert not self.decode_manager.runnable, "rebuild requires no running decode"
        torch.cuda.synchronize(self.device)
        if self.config.tp_info.size > 1:
            self.sync_all_ranks()
        self.engine.rebuild_runtime_cache(
            moe_cache_size=moe_cache_size, num_pages=num_pages, num_mamba_slots=num_mamba_slots,
            num_swa_pages=num_swa_pages,
        )
        if num_pages is not None or num_mamba_slots is not None or num_swa_pages is not None:
            # Any of these resizes invalidates the prefix cache: a KV resize leaves stale page
            # indices, a mamba resize leaves stale GDN-snapshot slot ids, and a window-pool resize
            # (num_swa_pages) reallocates the SWA/window token pool, leaving stale slot ids in the
            # radix tree. Rebuild the prefix cache + reclaim the resized free-lists.
            self.cache_manager.rebuild(self.engine.num_pages, self.engine.page_table)
            if num_pages is not None:
                # token_pool is sized to the page table; only a KV-page resize reallocates it.
                # A mamba-only rebuild leaves the page table untouched, so skip this (else it
                # needlessly reallocates + zeros the whole GPU token_pool every mamba resize).
                self.table_manager.rebuild(self.engine.page_table)
                self.token_pool = self.table_manager.token_pool
            self.cache_manager.check_integrity()
        # The prefill chunk cap tracks the CURRENT window-pool size (DSV4); a rebuild that
        # shrank the pool must shrink the cap too, or the next long prompt is chunked against
        # the stale budget and crashes _alloc_window.
        _chunk_cap = self.cache_manager.prefill_chunk_budget
        self.prefill_budget = (
            min(self.config.max_extend_tokens, _chunk_cap)
            if _chunk_cap else self.config.max_extend_tokens
        )
        if self.config.tp_info.size > 1:
            self.sync_all_ranks()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        # Expose the un-drained batch to _process_one_msg (abort in-flight check). Assigning
        # before the message loop is what makes the check airtight: the batch launched later
        # this iteration can only be probed by messages of the NEXT iteration, which sees it here.
        self._last_data = last_data
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None  # a queued rebuild to drain toward + execute
            or self._kv_wait  # an idle grow has to run; blocking here would wait for another request
            or self._pending_flush is not None  # a deferred flush to run at idle or time out
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Execute a queued cache rebuild once the scheduler is fully idle (the safe point):
        # no last batch to process, no pending prefill, no running decode. finished_reqs is
        # NOT a gate — those requests are already freed (no live GPU/page resources).
        if self._pending_rebuild is not None and last_data is None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            self._execute_pending_rebuild()
        if last_data is None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            self._maybe_grow_kv()
        self._check_pending_flush()

        # Order this iteration's host->device token_pool copies (issued on ``self.stream``
        # during scheduling) after the previous batch's sampled-token writes (issued on the
        # engine stream in ``_forward``). Without this, a request that reuses a just-freed
        # table_idx can have its freshly copied prompt clobbered by the prior occupant's
        # still-pending output write -- corrupting tokens (e.g. dropping an image
        # placeholder, which the multimodal merge then rejects).
        self.stream.wait_stream(self.engine.stream)
        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                # COW-restore GDN snapshots for prefix hits ON THE ENGINE STREAM, after the
                # cross-stream wait and before the forward reads the live slot (program order
                # vs the prior batch's snapshot writes). Doing this on self.stream would race.
                self._restore_linear_states(forward_input.batch)
                ongoing_data = (forward_input, self._forward(forward_input))

        # The drain issues GPU-visible writes to state the batch just launched still reads: the
        # page-table re-point and, for the paged-SWA pools, the full->swa (DSV4: full->window)
        # sentinel scatter. DSV4 stages the page table at replay time and translates
        # full_to_window INSIDE the captured graph, so an unordered drain can redirect an
        # in-flight forward. copy_done only covers batch N; order against N+1 explicitly.
        self.stream.wait_stream(self.engine.stream)
        self._process_last_data(last_data)
        self._flush_abort_acks()
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (
            self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None  # a queued rebuild to execute at idle
            or self._kv_wait
            or self._pending_flush is not None  # a deferred flush to run at idle or time out
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Non-overlap mode has no last_data to drain; execute a queued rebuild as soon as
        # the scheduler is idle (no pending prefill / running decode). Without this, a
        # rebuild in DISABLE_OVERLAP_SCHEDULING mode stays pending until the HTTP timeout.
        if self._pending_rebuild is not None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            self._execute_pending_rebuild()
        if not (self.prefill_manager.runnable or self.decode_manager.runnable):
            self._maybe_grow_kv()
        self._check_pending_flush()

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            # already inside engine_stream_ctx (run_forever); restore on the engine stream
            self._restore_linear_states(forward_input.batch)
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)
        self._flush_abort_acks()

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        # DSV4 (owned-KV) decode reads its per-token window/cmp/idx slot maps off the attention
        # backend's per-batch SNAPSHOT (staged in prepare_for_replay right before the replay, on
        # the same stream, like the generic out_loc copy_from), not the live slot maps -- so the
        # next batch's allocate_paged cannot corrupt the in-flight graph replay. DSV4 overlaps.
        if ENV.DISABLE_OVERLAP_SCHEDULING or self.spec is not None:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _wait_copy_done(
        self, copy_done: torch.cuda.Event, batch: Batch, timeout: float = 120.0
    ) -> None:
        """Synchronize copy_done with an operational step timeout.

        Avoids indefinite CPU spin-wait during GPU or NCCL collective hangs.
        If timeout is exceeded, sends ErrorReplyMsg to abort the affected requests
        and raises TimeoutError to trigger a clean supervisor restart.
        """
        if copy_done.query():
            return

        start_time = time.monotonic()
        while not copy_done.query():
            elapsed = time.monotonic() - start_time
            if elapsed > timeout:
                logger.error_rank0(
                    f"Forward step execution timed out after {elapsed:.1f}s (operational limit: {timeout}s)! "
                    f"Possible GPU desync or NCCL collective deadlock across {len(batch.reqs)} active request(s)."
                )
                error_replies = []
                for req in batch.reqs:
                    logger.error_rank0(
                        f"Dropping timed-out request {req.uid} (prompt_len={len(req.input_ids)}, "
                        f"output_len={len(req.output_ids)})."
                    )
                    error_replies.append(
                        ErrorReplyMsg(
                            uid=req.uid,
                            error=f"GPU forward execution timed out after {elapsed:.1f}s (operational limit: {timeout}s).",
                            code="timeout",
                        )
                    )
                    try:
                        self._free_req_resources(req)
                    except Exception as e:
                        logger.warning(f"Error freeing resources for req {req.uid}: {e}")

                if error_replies:
                    try:
                        self.send_result(error_replies)
                    except Exception as e:
                        logger.warning(f"Failed to send error replies for timed-out batch: {e}")

                raise TimeoutError(
                    f"GPU forward execution timed out after {elapsed:.1f}s (operational timeout: {timeout}s)."
                )

            # Yield CPU slightly after 5ms to avoid 100% CPU spinning while waiting for GPU
            if elapsed > 0.005:
                time.sleep(0.001)

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, output = last_data[0].batch, last_data[1]
        _, next_tokens_cpu, copy_done = output[:3]
        # SpecForwardOutput is a drain drop-in whose index 3 is emitted counts, so
        # only the named field can tell the two apart.
        fault_counts_cpu = getattr(output, "fault_counts_cpu", None)
        timeout = getattr(self.config, "step_timeout", None) or ENV.STEP_TIMEOUT.value
        self._wait_copy_done(copy_done, batch, timeout=timeout)
        # tokens are on the host from here: the per-request timings' token stamps
        now = time.perf_counter()
        reply: List[DetokenizeMsg] = []
        error_replies: List[ErrorReplyMsg] = []
        new_finished_reqs: Set[Req] = set()
        generated = None
        with self.cache_manager.lazy_free_region():
            if batch.phase == "verify":
                generated = self._drain_verify(batch, output, reply, new_finished_reqs)
            else:
                for i, req in enumerate(batch.reqs):
                    if isinstance(req, ChunkedReq):
                        # Don't cache intermediate chunks; the full prompt is cached once when the
                        # final chunk is processed. Caching here snapshots a handle the next chunk
                        # already copied (overlap), so cache_req double-frees the prior chunk.
                        if req.aborted:
                            # Aborted mid-chunked-prefill while this chunk was in flight: the abort
                            # popped the pending continuation (no next chunk launches), and this
                            # drain point frees the chunk's pages/slots exactly once.
                            self._free_req_resources(req)
                        else:
                            # The chunk's PAGES must stay out of the tree (double-free above), but its
                            # x64 GDN checkpoint owns none: archive it here, stream-ordered after it.
                            self.cache_manager.archive_chunk_track(req)
                        continue
                    if req.aborted:
                        # Aborted while this final-chunk prefill / decode step was in flight: free
                        # here (the forward is drained) and finish the request. No DetokenizeMsg --
                        # the abort ack flushed after this method stays the uid's terminal reply.
                        self.decode_manager.remove_req(req)
                        self._free_req_resources(req)
                        new_finished_reqs.add(req)
                        continue
                    if req in self.finished_reqs:
                        # Overlap scheduling launched one more decode step for a request that
                        # already terminated (filter_reqs keeps it while output budget remains,
                        # and the next batch is scheduled before this drain runs). Its resources
                        # are freed below/already; shipping this token would append past the
                        # client's terminal reply.
                        continue
                    fault = (
                        row_fault(fault_counts_cpu[i]) if fault_counts_cpu is not None else None
                    )
                    if fault is not None and fault.is_fault:
                        # The sampler cannot tell such a row from a healthy one: argmax answers
                        # index 0 and the inverse-CDF draw leaves its zero-filled out untouched,
                        # so this request would stream token 0 ("!") to its output budget.
                        # Quarantine it as well as failing it: its live slot holds the NaN and its
                        # pages the step that produced it, and neither may become a reusable prefix.
                        req.generation_poisoned = True
                        error_replies.append(
                            self._degenerate_generation_reply(
                                req,
                                batch,
                                code=NONFINITE_LOGITS,
                                detail=f"non-finite logits ({fault.describe()}) at length "
                                f"{req.input_ids.numel()}",
                            )
                        )
                        if req not in self.finished_reqs:
                            self.decode_manager.remove_req(req)
                            self._free_req_resources(req)
                            new_finished_reqs.add(req)
                        continue
                    msg, _ = self._commit_tokens(req, next_tokens_cpu[i : i + 1])
                    token = int(next_tokens_cpu[i].item())
                    if note_repeated_token(req, token, ENV.STORM_TOKEN_RUN.value):
                        # The backstop for a collapse that stayed finite. No terminal token
                        # reply: the client has already been streamed the run, so the request
                        # ends as an error rather than as a completion of nonsense.
                        error_replies.append(
                            self._degenerate_generation_reply(
                                req,
                                batch,
                                code=DEGENERATE_REPEAT,
                                detail=f"token {token} repeated {req.repeat_run}x in a row",
                            )
                        )
                        if req not in self.finished_reqs:
                            self.decode_manager.remove_req(req)
                            self._free_req_resources(req)
                            new_finished_reqs.add(req)
                        continue
                    req.stats.stamp_token(now)
                    reply.append(msg)
                    # a timings_per_token request gets the snapshot on every reply, growing
                    if msg.finished or req.sampling_params.timings_per_token:
                        msg.timings = req.stats.finish(req.input_ids.numel() - req.prompt_len)
                    finished = msg.finished

                    # NOTE: overlap scheduling may make the request freed twice, skip second free
                    if finished and req not in self.finished_reqs:
                        self.decode_manager.remove_req(req)
                        self._free_req_resources(req)
                        new_finished_reqs.add(req)
                    elif batch.is_prefill and req.table_idx != -1:
                        # for prefill, non-chunk req, cache the prefix.
                        # Polymorphic: the DSV4 naive manager keeps the request's slots (no-op);
                        # the generic manager inserts the prefix into its radix/naive cache.
                        # table_idx == -1 is defense-in-depth: aborts mark in-flight requests
                        # instead of freeing them (handled above), so a freed request should
                        # never reach this commit -- but if a future path frees one early, skip
                        # rather than re-read the freed page-table row (and on hybrid, deref the
                        # None'd GDN ping-pong slots).
                        self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        # Stamp each reply with the post-batch KV page occupancy so the frontend (shell
        # status bar) can show live KV usage without a separate query.
        used, total = self._kv_usage_pages()
        mamba_slots = self._mamba_slot_usage()
        swa_tokens = self._swa_token_usage()
        if reply:
            mem = self._gpu_mem_bytes()
            mamba_used, mamba_total = mamba_slots or (0, 0)
            swa_used, swa_total = swa_tokens or (0, 0)
            for m in reply:
                m.kv_used_pages = used
                m.kv_total_pages = total
                m.mamba_used_slots = mamba_used
                m.mamba_total_slots = mamba_total
                m.swa_used_tokens = swa_used
                m.swa_total_tokens = swa_total
                m.gpu_mem_bytes = mem
        self.status_reporter.report_batch(
            batch,
            running_reqs=len(self.decode_manager.running_reqs),
            queue_reqs=len(self.prefill_manager.pending_list),
            kv_used_pages=used,
            kv_total_pages=total,
            page_size=self.config.page_size,
            mamba_slots=mamba_slots,
            swa_tokens=swa_tokens,
            generated=generated,
        )
        self.send_result(reply)
        if error_replies:
            self.send_result(error_replies)
        self._storm_check_fatal()

    def _storm_check_fatal(self) -> None:
        """Exit the server while a poisoned engine is still answering, rather than after it stops.

        Two rules, weighed apart, because the two faults are not equally believable as one-offs.

        * A non-finite logits row is arithmetic over state that is already bad where the sampler
          reads it, so the quarantine has demonstrably not recovered the engine: the next request
          that touches the same pages dies at the same offset. Observed 2026-11-10 -- two trips
          five seconds apart, then nine hours of a live port answering nothing, because the client
          had given up retrying and a rule that counts trips across a window could never fire.
          One such trip is therefore fatal on its own by default.
        * A repeated-token collapse stayed finite, and a model can genuinely repeat itself, so it
          only counts toward the generic limit shared by every breaker.

        Either way the instance has to go down: a new process is the only thing that has ever
        cleared engine state, and a fast visible exit beats a server that answers nothing.
        """
        limit = int(ENV.STORM_FATAL_TRIPS.value)
        nonfinite_limit = int(ENV.STORM_FATAL_NONFINITE_TRIPS.value)
        if limit <= 0 and nonfinite_limit <= 0:
            return
        ledger = getattr(self, "_storm", None)
        if ledger is None:
            return
        window = float(ENV.STORM_FATAL_WINDOW_S.value)
        reasons: list[str] = []
        if nonfinite_limit > 0:
            bad_nonfinite = ledger.distinct_uids_within(window, code=NONFINITE_LOGITS)
            if bad_nonfinite >= nonfinite_limit:
                reasons.append(
                    f"{bad_nonfinite} distinct request(s) tripped the non-finite logits breaker "
                    f"in {window:.0f}s (limit {nonfinite_limit})"
                )
        if limit > 0:
            bad = ledger.distinct_uids_within(window)
            if bad >= limit:
                reasons.append(
                    f"{bad} distinct requests tripped a breaker in {window:.0f}s (limit {limit})"
                )
        if not reasons:
            return
        detail = "; ".join(reasons) + "; quarantine did not recover the engine"
        # Evidence first: the ranks are about to die and a pane's scrollback is not evidence.
        record_storm_event(
            ENV.STORM_EVENT_LOG.value,
            {
                "event": "storm_fatal_exit",
                "detail": detail,
                "trips": [
                    {"uid": uid, "code": code, "at": trip_at}
                    for trip_at, uid, code in ledger.recent(window)
                ],
                "total_trips": ledger.total,
            },
        )
        message = (
            f"{detail}. Exiting so the scheduler state holding the poison is discarded; "
            "set FREETOKEN_STORM_FATAL_NONFINITE_TRIPS=0 to keep serving through a non-finite "
            "row (and FREETOKEN_STORM_FATAL_TRIPS=0 for the other breakers), "
            "FREETOKEN_NAN_TRACE=1 on the next launch to name the module that produces it."
        )
        try:
            logger.error_rank0(f"STORM FATAL: {message}")
        except Exception:
            pass
        # Let the reply sender drain: this rank exits the moment we raise, and a client that got
        # no terminal reply would sit in wait_for_ack until ACK_IDLE_TIMEOUT instead of failing now.
        time.sleep(0.5)
        raise StormUnrecoverable(message)

    def _degenerate_generation_reply(
        self, req: Req, batch: Batch, *, code: str, detail: str
    ) -> ErrorReplyMsg:
        """Report one breaker trip and build the request's terminal error reply.

        Nothing in here may raise. It runs in the drain of a live TP4 step, and an
        exception escaping it takes down every scheduler rank and with them the API
        server -- a fault report that costs more than the request it reports is worse
        than the fault. Evidence therefore goes to the event log *before* the log call:
        a pane's scrollback is not evidence, and the first trip ever lost its report to
        an ``AttributeError`` raised by the logging call itself.
        """
        uid = int(req.uid)
        where = "NAN_TRACE off"
        try:
            tracer = getattr(self.engine, "nan_tracer", None)
            if tracer is not None:
                where = tracer.describe()
        except Exception as e:
            where = f"tracer unavailable ({type(e).__name__}: {e})"
        padded = getattr(batch, "padded_reqs", None)
        try:
            record = {
                "event": code,
                "uid": uid,
                "detail": detail,
                "tokens": int(req.input_ids.numel()),
                "phase": "prefill" if batch.is_prefill else "decode",
                "batch": int(batch.size),
                "padded_batch": int(len(padded)) if padded is not None else -1,
                "layer_major": bool(getattr(batch, "layer_major", False)),
                "table_idx": int(req.table_idx),
                "linear_slot": int(getattr(req, "linear_slot_idx", -1) or -1),
                "first_nonfinite_module": where,
            }
        except Exception as e:
            record = {"event": code, "uid": uid, "detail": detail, "context_error": repr(e)}
        try:
            record_storm_event(ENV.STORM_EVENT_LOG.value, record)
        except Exception as e:
            logger.warning(f"Could not record the breaker trip for req {uid}: {e!r}")
        try:
            ledger = getattr(self, "_storm", None)
            if ledger is None:
                ledger = self._storm = StormLedger()
            ledger.note(uid, code)
        except Exception as e:  # noqa: BLE001 -- accounting never costs more than its fault
            logger.warning(f"Could not ledger the breaker trip for req {uid}: {e!r}")
        message = (
            f"req {uid}: {detail}; first non-finite module: {where}; aborting the request "
            "instead of streaming a continuation that cannot become useful "
            "(FREETOKEN_ABORT_ON_NONFINITE_LOGITS=0 / FREETOKEN_STORM_TOKEN_RUN=0 disable)"
        )
        try:
            logger.error_rank0(message)
        except Exception:
            try:
                logger.warning(f"[rank0] {message}")
            except Exception:
                pass
        return ErrorReplyMsg(
            uid=uid,
            error=(
                f"FreeToken aborted this request: {detail}. The continuation could not have "
                "become useful, and the sampler cannot distinguish such a row from a healthy "
                "one, so it was failed rather than streamed."
            ),
            code=code,
        )

    def _drain_verify(self, batch: Batch, output, reply: List[DetokenizeMsg], new_finished_reqs: Set[Req]) -> int:
        """Commit each request's accepted tokens, return the slots its rejected rows took, then the target state; returns the tokens committed."""
        k = self.config.decode_rows_per_req - 1
        counts, finished = [], []
        now = time.perf_counter()
        for i, req in enumerate(batch.reqs):
            # spec forces normal_loop, which frees an aborted request between steps; overlap would need rollback_tail here and must not donate the GDN slot the verify rolled
            assert not req.aborted, "a verify batch drained an aborted request"
            cached = req.cached_len
            emitted = int(output.emitted_cpu[i])
            msg, n = self._commit_tokens(req, output.next_tokens_cpu[i, : emitted + 1])
            # k drafts were proposed; the step hands _commit_tokens the emitted accepted
            # drafts plus the target's own token, so when a stop lands on a draft every
            # committed token is an accepted draft: count min(n, emitted), with n already
            # post stop-truncation. Never exceeds predicted_n - 1: the request's first
            # committed token came from prefill, outside every verify step.
            req.stats.stamp_token(now)
            req.stats.draft_n += k
            req.stats.draft_n_accepted += min(n, emitted)
            reply.append(msg)
            # a timings_per_token request gets the snapshot on every reply, growing
            if msg.finished or req.sampling_params.timings_per_token:
                msg.timings = req.stats.finish(req.input_ids.numel() - req.prompt_len)
            req.commit_pending(n)
            self.cache_manager.rollback_tail(req, req.cached_len, cached + k + 1)
            counts.append(n)
            if msg.finished:
                finished.append(req)
        # before any finish free: the hybrid finish donates the live GDN slot the commit writes
        self.engine.commit_verify(batch.reqs, counts, output.next_tokens_gpu)
        # the last committed token is pending at cached_len, like a decode step's sampled token
        table, pos, last = torch.tensor(
            [(r.table_idx, r.cached_len, n - 1) for r, n in zip(batch.reqs, counts)],
            dtype=torch.int64, pin_memory=self.device.type == "cuda",
        ).to(self.device, non_blocking=True).unbind(1)
        self.token_pool[table, pos] = output.next_tokens_gpu.gather(1, last.unsqueeze(1)).squeeze(1).to(self.token_pool.dtype)
        for req in finished:
            self.decode_manager.remove_req(req)
            self._free_req_resources(req)
            new_finished_reqs.add(req)
        return sum(counts)

    def _commit_tokens(self, req: Req, tokens: torch.Tensor) -> Tuple[DetokenizeMsg, int]:
        """Append sampled tokens in order until one finishes the request; returns the reply and how many tokens were committed."""
        params = req.sampling_params
        committed: List[int] = []
        finished, finish_reason, matched_stop = False, None, None
        for j in range(tokens.numel()):
            req.append_host(tokens[j : j + 1])
            token = int(tokens[j].item())
            committed.append(token)
            # EOS / stop-string -> "stop", output budget exhausted -> "length";
            # EOS and stop strings win over length.
            # Overlap can advance device_len one token ahead of this host reply.
            hit_length = req.input_ids.numel() >= req.max_device_len
            hit_eos = not params.ignore_eos and token in self.eos_token_ids
            matched_stop = self._match_stop_str(req) if not hit_eos and params.stop_strs else None
            if hit_length or hit_eos or matched_stop is not None:
                finished = True
                finish_reason = "stop" if (hit_eos or matched_stop is not None) else "length"
                break
            if token == self.toolcall_anchor_id and req.toolcall_anchor_len is None:
                req.toolcall_anchor_len = req.input_ids.numel()
        msg = DetokenizeMsg(
            uid=req.uid,
            next_tokens=committed,
            finished=finished,
            finish_reason=finish_reason,
            matched_stop=matched_stop,
            stop_strs=params.stop_strs or None,
        )
        return msg, len(committed)

    def _match_stop_str(self, req: Req) -> str | None:
        """First stop string present in this request's generated tail, else None. Decodes
        only a short suffix (bounded by the longest stop string's char length, so a stop of
        N chars spans at most N tokens) to keep the per-step cost small."""
        stop_strs = req.sampling_params.stop_strs
        prompt_len = req.max_device_len - req.output_len
        if len(req.input_ids) <= prompt_len:
            return None
        max_chars = max(len(s) for s in stop_strs)
        tail_start = max(prompt_len, len(req.input_ids) - (max_chars + 1))
        tail = self.tokenizer.decode(req.input_ids[tail_start:].tolist())
        for s in stop_strs:
            if s in tail:
                return s
        return None

    def _kv_usage_pages(self) -> Tuple[int, int]:
        """(used_pages, total_pages) of the KV page pool.

        ``used`` follows SGLang's logging semantics: allocated pages that are not
        evictable (active requests + protected prefix cache). Evictable prefix-cache
        pages are available to future requests, so they are excluded from usage.
        Always the manager's own primary pool (for DSV4 the FULL cmp/idx tier); the
        window (swa) tier is reported separately by ``_swa_token_usage``.
        """
        return self.cache_manager.page_usage()

    def _mamba_slot_usage(self) -> Tuple[int, int] | None:
        """(used_slots, total_slots) of the GDN-state (mamba) pool for hybrid models, else None.

        Mirrors SGLang's mamba-pool semantics: ``total`` excludes the reserved padding
        sink (slot 0); ``used`` excludes free slots and evictable tree snapshots.
        """
        if not self.cache_manager.is_hybrid:
            return None
        total = self.cache_manager.linear_state_pool.num_slots - 1
        return total - self.cache_manager.mamba_available_size, total

    def _swa_token_usage(self) -> Tuple[int, int] | None:
        """(used_tokens, total_tokens) of the window (swa) pool for SWA models, else None.

        Mirrors the mamba accounting: ``total`` excludes the pool's reserved sentinel
        unit; ``used`` excludes free slots and evictable (unlocked) tree tokens.
        """
        cm = self.cache_manager
        if not cm.swa_paged:
            return None
        total = cm.swa_pool.swa_num_tokens - 1
        return total - cm.swa_available_size, total

    def _gpu_mem_bytes(self) -> int:
        """Bytes this engine process holds on the GPU (torch's reserved caching-allocator
        pool: weights + KV + MoE cache + graphs). 0 on CPU. Cheap, no device sync."""
        if self.device.type != "cuda":
            return 0
        return torch.cuda.memory_reserved(self.device)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            tombstones = getattr(self, "_abort_tombstones", None)
            if tombstones is not None and msg.uid in tombstones:
                tombstones.pop(msg.uid, None)
                logger.debug_rank0(
                    "Dropping request %d because its abort arrived before admission", msg.uid
                )
                return
            if msg.mm_items and self.engine.encoder_cache is None:
                # no encoder runtime: fail loudly instead of decoding unexpanded placeholders
                self.send_result(
                    [
                        ErrorReplyMsg(
                            uid=msg.uid,
                            error="image input is not supported by this server",
                        )
                    ]
                )
                return
            if (
                msg.mm_items
                and self.engine.config.mm.encoder_out_of_process
                and any(item.precomputed_embeddings is None for item in msg.mm_items)
            ):
                # the TP ranks hold no tower in this mode; refusing here keeps every rank's forward alive
                self.send_result(
                    [
                        ErrorReplyMsg(
                            uid=msg.uid,
                            error="image input reached the engine without its CPU encoding",
                        )
                    ]
                )
                return
            input_len = len(msg.input_ids)
            seq_limit = self._kv_seq_limit()
            max_output_len = seq_limit - input_len
            if max_output_len <= 0:
                logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {seq_limit}, "
                    f"request {msg.uid} is dropped."
                )
                # Tell the client instead of dropping silently — otherwise its wait_for_ack
                # never sees a `finished` reply and hangs until the request times out.
                self.send_result(
                    [
                        ErrorReplyMsg(
                            uid=msg.uid,
                            # "prompt is too long: N tokens > M" is the phrasing Claude Code and
                            # OpenClaw match on; the Anthropic wire has no error code to read.
                            error=(
                                f"prompt is too long: {input_len} tokens > {seq_limit} maximum "
                                f"(prompt + generation); shorten the prompt or increase the KV "
                                f"cache budget"
                            ),
                            # OpenAI's standard class for this, for clients that read a code.
                            code="context_length_exceeded",
                        )
                    ]
                )
                return
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            ek = getattr(self.engine, "elastic_kv", None)
            if ek is not None:
                need = self._aligned_kv_need(input_len, msg.sampling_params.max_tokens, ek.page_size)
                current = self.engine.num_pages * ek.page_size
                if need > current:
                    self._kv_wait.append(msg)
                    logger.info_rank0(
                        f"elastic KV: holding request {msg.uid} "
                        f"({need} tokens > pool {current}, max {ek.max_tokens})"
                    )
                    return
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            self._kv_wait = [waiting for waiting in self._kv_wait if waiting.uid != msg.uid]
            tombstones = getattr(self, "_abort_tombstones", None)
            if tombstones is None:
                tombstones = self._abort_tombstones = {}
            tombstones[msg.uid] = None
            # Unknown aborts normally consume their tombstone when the cross-worker UserMsg
            # catches up. Bound hostile/no-followup abort traffic without affecting realistic
            # in-flight concurrency.
            while len(tombstones) > 65_536:
                tombstones.pop(next(iter(tombstones)))
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if (
                req_to_free is not None
                and req_to_free.mm_items
                and self.engine.encoder_cache is not None
            ):
                # drop the aborted request's claims; entries it held alone die here
                self.engine.encoder_cache.release(
                    msg.uid, [item.hash for item in req_to_free.mm_items]
                )
            if req_to_free is not None:
                # SGLang-style abort: never free resources under an in-flight forward. If the
                # request is in the launched-but-not-drained batch (overlap), only mark it;
                # _process_last_data frees it this same iteration, after copy_done.synchronize()
                # -- so its KV pages / GDN slots are never recycled mid-write, and the
                # finished=False prefix-commit can't run on a freed request. A request with no
                # forward in flight (e.g. a decode req starved behind a long chunked prefill)
                # is freed immediately -- deferring would leak until its next batch, which
                # strict prefill-priority puts arbitrarily far away.
                inflight = (
                    self._last_data is not None
                    and req_to_free in self._last_data[0].batch.reqs
                )
                if inflight:
                    req_to_free.aborted = True
                else:
                    self._free_req_resources(req_to_free)
            # Always acknowledge the abort, even when the request already left the manager,
            # but NOT yet: overlap_loop still has to publish the prior forward's sampled reply.
            # _flush_abort_acks runs after _process_last_data, making this a true terminal
            # accounting barrier for FrontendManager/prepare-stop.
            self._pending_abort_acks.add(msg.uid)
        elif isinstance(msg, CacheRebuildBackendMsg):
            # v1 scope: only if_idle, single-rank, non-owned-KV. drain mode and TP rebuild
            # need the drain-gate / all-rank failure-agreement machinery (deferred), so we
            # reject them cleanly rather than ship hang-prone half-wired paths.
            if not self.cache_manager.supports_runtime_rebuild:
                self._reply_rebuild(
                    msg.request_id, "unsupported", "this model's cache does not support runtime rebuild"
                )
            elif msg.mode != "if_idle":
                self._reply_rebuild(
                    msg.request_id, "unsupported", f"mode {msg.mode!r} unsupported (use if_idle)"
                )
            elif self.config.tp_info.size > 1:
                self._reply_rebuild(
                    msg.request_id, "unsupported", "runtime rebuild unsupported under TP > 1"
                )
            elif self.prefill_manager.runnable or self.decode_manager.runnable:
                # if_idle: refuse rather than wait. (finished_reqs hold no resources — they
                # are already freed — so they do not block a rebuild.)
                self._reply_rebuild(msg.request_id, "busy")
            else:
                self._pending_rebuild = msg
        elif isinstance(msg, FlushCacheBackendMsg):
            self._handle_flush(msg)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _restore_linear_states(self, batch) -> None:
        """COW-restore a hybrid prefix hit's GDN snapshot into its freshly-allocated live slot
        (first chunk only). MUST run on the ENGINE stream so it is program-ordered after the
        prior batch's snapshot writes and before this forward reads the live slot."""
        pool = self.engine.linear_state_pool
        if pool is None or not batch.is_prefill:
            return
        for req in batch.reqs:
            src, host = req.mamba_restore_src, req.mamba_restore_host
            # Consumed exactly once: a stale ref would re-copy over a live slot mid-decode.
            req.mamba_restore_src = req.mamba_restore_host = None
            fresh = req.linear_state_fresh
            req.linear_state_fresh = False
            if src is not None:
                pool.copy_from(src, req.linear_slot_idx)
            elif host is not None:
                # Same COW-restore one tier up -- the checkpoint outlived its device slot in
                # pinned RAM, which is what makes an intermediate chunk's boundary resumable.
                pool.from_host(host, req.linear_slot_idx)
                logger.info_rank0(
                    f"GDN bank: resuming req {req.uid} from host slot {host}")
            elif fresh and req.linear_slot_idx is not None and ENV.LINEAR_STATE_CLEAR_ON_ADMIT.value:
                # Nothing is being restored into this slot and the GDN kernels read it as
                # carry-in, so it still holds whoever had it before -- possibly their NaN. Only a
                # freshly admitted request gets here: a chunked continuation inherits live state
                # and is never marked fresh.
                pool.clear_slots([req.linear_slot_idx])

    def _free_req_resources(self, req: Req) -> None:
        # Idempotent: an EOS-finished request can stay in running_reqs (output budget left), so an
        # abort in the same overlap iteration races _process_last_data and would free it twice --
        # double-freeing its table_idx and (hybrid) GDN slots onto the free-list, handing the same
        # slots to two later requests. table_idx == -1 marks an already-freed request.
        if req.table_idx == -1:
            return
        if self.spec is not None:
            self.spec.release(req)
        # Polymorphic free: the DSV4 manager returns the request's window pages + cmp/idx blocks
        # to their tier free-lists; the generic manager frees its KV pages (it reads
        # page_table[req.table_idx], so free the table entry after).
        self.cache_manager.cache_req(req, finished=True)
        self.table_manager.free(req.table_idx)
        req.table_idx = -1

    def _reply_rebuild(self, request_id: str, status: str, error: str | None = None) -> None:
        # Single source of truth with the rollback snapshot (_current_cache_geometry): mamba is
        # usable slots (padding sink excluded, matching the status-bar gauge), and num_swa_pages
        # reports 0 unless the model actually has a window pool.
        geo = self._current_cache_geometry()
        self.send_result(
            [
                CacheRebuildResultMsg(
                    request_id=request_id,
                    status=status,
                    moe_cache_size=geo["moe_cache_size"] or 0,
                    num_pages=geo["num_pages"],
                    mamba_slots=geo["num_mamba_slots"] or 0,
                    num_swa_pages=geo["num_swa_pages"] or 0,
                    error=error,
                )
            ]
        )

    def _kv_seq_limit(self) -> int:
        """Admission ceiling: the elastic max when it is configured, otherwise the live pool."""
        ek = getattr(self.engine, "elastic_kv", None)
        limit = self.engine.max_seq_len if ek is None else ek.max_tokens
        return limit - self.prefill_manager.token_headroom

    def _aligned_kv_need(self, input_len: int, max_tokens: int, page_size: int) -> int:
        return div_ceil(input_len + max_tokens + self.prefill_manager.token_headroom, page_size) * page_size

    def _maybe_grow_kv(self) -> None:
        """When idle, grow the KV pool to cover the parked requests and shrink the MoE cache.

        Every rank runs this on the same parked list, then ``rebuild_cache`` barriers, so TP
        stays in lockstep. Requests that fit the current pool were never parked, so they are
        not stuck behind a longer one.
        """
        from freetoken.engine.cache_budget import elastic_grow_tokens

        ek = getattr(self.engine, "elastic_kv", None)
        if ek is None or not self._kv_wait:
            return
        if self.prefill_manager.runnable or self.decode_manager.runnable:
            return
        needs = [
            self._aligned_kv_need(len(msg.input_ids), msg.sampling_params.max_tokens, ek.page_size)
            for msg in self._kv_wait
        ]
        current = self.engine.num_pages * ek.page_size
        target = elastic_grow_tokens(current, needs, ek.max_tokens)
        if target is not None:
            step = div_ceil(32_768, ek.page_size) * ek.page_size
            target = min(ek.max_tokens, max(target, current + step))
            # Layer-major (<=32k) also keeps a fixed scratch reserve. Longer prompts only
            # pay for the extra KV pages, which is how 14500 slots served a 200k pool.
            activation_tokens = max(len(msg.input_ids) for msg in self._kv_wait)
            try:
                moe, pages, _overlap = ek.plan(target, activation_tokens=activation_tokens)
            except ValueError as exc:
                logger.error(f"elastic KV grow rejected: {exc}")
                msg = self._kv_wait.pop(0)
                self._reject_context(msg, ek.max_tokens)
                return
            logger.info_rank0(
                f"elastic KV: growing pool {current} -> {pages * ek.page_size} tokens, "
                f"MoE slots -> {moe} (prompt {activation_tokens} tokens)"
            )
            self.rebuild_cache(moe_cache_size=moe, num_pages=pages)
            torch.cuda.empty_cache()
            current = self.engine.num_pages * ek.page_size
        self._admit_kv_wait(current, ek.max_tokens, ek.page_size)

    def _admit_kv_wait(self, capacity: int, kv_max: int, page_size: int) -> None:
        kept: list[UserMsg] = []
        reserved = 0
        for msg in self._kv_wait:
            need = self._aligned_kv_need(len(msg.input_ids), msg.sampling_params.max_tokens, page_size)
            if need > kv_max:
                self._reject_context(msg, kv_max)
                continue
            if reserved + need <= capacity:
                self.prefill_manager.add_one_req(msg)
                reserved += need
            else:
                kept.append(msg)
        self._kv_wait = kept

    def _reject_context(self, msg: UserMsg, limit: int) -> None:
        input_len = len(msg.input_ids)
        logger.warning_rank0(
            f"Input sequence length {input_len} exceeds {limit}, request {msg.uid} is dropped."
        )
        self.send_result(
            [
                ErrorReplyMsg(
                    uid=msg.uid,
                    error=(
                        f"prompt is too long: {input_len} tokens > {limit} maximum "
                        f"(prompt + generation); shorten the prompt or increase the KV "
                        f"cache budget"
                    ),
                    code="context_length_exceeded",
                )
            ]
        )

    def _execute_pending_rebuild(self) -> None:
        from freetoken.engine.engine import CacheRebuildRejected

        msg = self._pending_rebuild
        assert msg is not None
        self._pending_rebuild = None
        requested = {
            "moe_cache_size": msg.moe_cache_size,
            "num_pages": msg.num_pages,
            "num_mamba_slots": msg.num_mamba_slots,
            "num_swa_pages": msg.num_swa_pages,
        }
        # Rollback target: the CURRENT (serving) sizes of ONLY the pools this request touches.
        # Passing the untouched pools too would trip rebuild_cache's KV/mamba/SWA gate and wipe
        # the prefix cache that a successful resize of just the requested pool preserves.
        snapshot = self._current_cache_geometry()
        prior = {k: snapshot[k] for k, v in requested.items() if v is not None}
        # Cleared here, set by engine.rebuild_runtime_cache at its point of no return — lets the
        # except below tell a pre-teardown failure (engine untouched) from a mid-teardown one.
        self.engine.rebuild_teardown_started = False
        try:
            self.rebuild_cache(**requested)
        except CacheRebuildRejected as e:
            # Rejected before any destructive free — old cache intact, keep serving.
            logger.warning(f"cache rebuild rejected: {e}")
            self._reply_rebuild(msg.request_id, "rejected", error=str(e))
            return
        except Exception as e:  # noqa: BLE001
            if not getattr(self.engine, "rebuild_teardown_started", True):
                # Failed before the destructive phase began: graphs and pools are untouched and
                # the engine is still serving. A destructive rollback would only add risk.
                logger.error(f"cache rebuild failed before teardown: {e!r} — old cache intact")
                self._reply_rebuild(msg.request_id, "rejected", error=repr(e))
                return
            if self.config.tp_info.size > 1:
                # A lone-rank failure cannot be rolled back symmetrically: rebuild_cache runs TP
                # barriers, and ranks that succeeded will not re-enter them — a solo rollback
                # would desync the group. Keep the latch-failed behavior for tp>1.
                logger.error(f"cache rebuild failed: {e!r} — tp>1, latching failed")
                self._reply_rebuild(msg.request_id, "failed", error=repr(e))
                return
            # The destructive phase failed — typically a CUDA OOM while reallocating a pool or
            # recapturing graphs. The graphs/pools are already torn down, so the engine cannot
            # serve as-is. Rather than latch "failed" (which forces a full process restart),
            # rebuild the touched pools back to the sizes that were serving a moment ago: they
            # fit before, so shrinking back frees the just-attempted allocation and restores
            # service. Only if the rollback ALSO fails is the engine genuinely wedged. (Post-OOM
            # CUDA state is not guaranteed sane — a rollback that succeeds here may still surface
            # a deferred fault on a later request; that residual risk is accepted over always
            # forcing a restart.)
            logger.error(f"cache rebuild failed: {e!r} — rolling back to the previous geometry")
            try:
                self.rebuild_cache(**prior)
            except Exception as e2:  # noqa: BLE001 — rollback failed too; genuinely unrecoverable
                logger.error(f"cache rebuild rollback failed: {e2!r} — server latched failed")
                self._reply_rebuild(
                    msg.request_id,
                    "failed",
                    error=f"{e!r}; rollback to the prior geometry also failed: {e2!r}",
                )
                return
            logger.warning("cache rebuild rolled back to the previous geometry — still serving")
            self._log_cache_geometry("Cache rolled back")
            self._reply_rebuild(
                msg.request_id, "rejected", error=f"rebuild failed and was rolled back: {e!r}"
            )
            return
        # Outside the try: an ack/send failure after a fully-applied rebuild must not be
        # mistaken for a rebuild failure and roll back the geometry the engine now serves.
        self._log_cache_geometry("Cache rebuilt")
        self._reply_rebuild(msg.request_id, "ok")

    def _handle_flush(self, msg: FlushCacheBackendMsg) -> None:
        # Same idle/timeout contract as sglang's /flush_cache. Even timeout=0 waits for this iteration's safe point.
        if self.config.tp_info.size > 1:
            # each rank checks the deadline on its own clock, so a deferred flush could run on only some ranks
            self._reply_flush(msg.request_id, "unsupported", "cache flush unsupported under TP > 1")
        elif self._pending_flush is not None:
            self._reply_flush(msg.request_id, "busy", "another cache flush is already waiting")
        else:
            self._pending_flush = (msg, time.monotonic() + msg.timeout)

    def _check_pending_flush(self) -> None:
        """Run the pending flush once the scheduler is fully idle, or fail it at its deadline."""
        if self._pending_flush is None:
            return
        msg, deadline = self._pending_flush
        if self._last_data is None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            self._pending_flush = None
            freed = self.cache_manager.flush()
            self.cache_manager.check_integrity()
            logger.info_rank0(f"Cache flushed: {freed} prefix-cache tokens freed")
            self._reply_flush(msg.request_id, "ok")
        elif time.monotonic() >= deadline:
            self._pending_flush = None
            error = (
                "scheduler is not idle: "
                f"{len(self.prefill_manager.pending_list)} queued, "
                f"{len(self.decode_manager.running_reqs)} running"
                + (", last batch still in flight" if self._last_data is not None else "")
            )
            logger.warning_rank0(f"Cache not flushed, {error}")
            self._reply_flush(msg.request_id, "timeout" if msg.timeout > 0 else "busy", error)

    def _reply_flush(self, request_id: str, status: str, error: str | None = None) -> None:
        self.send_result([FlushCacheResultMsg(request_id=request_id, status=status, error=error)])

    def _current_cache_geometry(self) -> dict:
        """The pools' current (serving) sizes as rebuild_cache kwargs — the rollback snapshot and
        the single source for _reply_rebuild's readout. None for a pool this model lacks
        (rebuild_cache skips those; the reply maps them to the wire format's 0). num_swa_pages is
        the CONCRETE current window (usable pages) so a rollback restores it byte-for-byte,
        whether it was pinned or ratio-derived."""
        eng = self.engine
        from freetoken.kvcache.cache_status import window_pool_spec

        spec = window_pool_spec(self.config)
        num_swa_pages = eng.kv_cache.window_pages if spec is not None else None
        return dict(
            num_pages=eng.num_pages,
            moe_cache_size=eng.moe_offload_cache.cache_size if eng.moe_offload_cache is not None else None,
            num_mamba_slots=(eng.linear_state_pool.num_slots - 1) if eng.linear_state_pool is not None else None,
            num_swa_pages=num_swa_pages,
        )

    def _log_cache_geometry(self, event: str) -> None:
        """One-line readout of every pool's new size + VRAM after a rebuild changed them:
        full KV always; swa/mamba/MoE only for models with the pool. Byte figures are
        best-effort (0 when a unit cost cannot be measured) and must never block the reply."""
        from freetoken.kvcache.cache_status import compute_cache_pools, compute_cache_unit_bytes

        try:
            pools = compute_cache_pools(self.engine)
            unit = compute_cache_unit_bytes(self.engine)
            kv_tokens = pools["num_pages"] * pools["page_size"]
            parts = [
                f"KV {pools['num_pages']} pages"
                f" ({kv_tokens} tokens, {_gib(kv_tokens * unit['kv_bytes_per_token'])})"
            ]
            if pools["num_swa_pages"]:
                swa_tokens = pools["num_swa_pages"] * pools["swa_page_size"]
                parts.append(
                    f"swa {pools['num_swa_pages']} pages"
                    f" ({swa_tokens} tokens, {_gib(swa_tokens * unit['swa_bytes_per_token'])})"
                )
            if pools["num_mamba_slots"]:
                parts.append(
                    f"mamba {pools['num_mamba_slots']} slots"
                    f" ({_gib(pools['num_mamba_slots'] * unit['mamba_bytes_per_slot'])})"
                )
            moe = self.engine.moe_offload_cache
            if moe is not None:
                parts.append(
                    f"MoE cache {moe.cache_size}/{moe.num_layers * moe.num_experts}"
                    f" ({_gib(moe.cache_size * unit['moe_bytes_per_expert'])})"
                )
            logger.info_rank0(f"{event}: " + ", ".join(parts))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"could not log cache geometry: {e!r}")

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        if self.spec is not None and batch.is_decode:
            batch.phase, batch.rows_per_req = "verify", self.config.decode_rows_per_req
        self.engine.graph_runner.pad_batch(batch)
        self._forward_iter += 1
        if not batch.is_prefill:
            # Free each decoding request's now-out-of-window SWA slots BEFORE the alloc below,
            # so they can back the new token -- this is what bounds the per-request swa
            # footprint during decode. (no-op unless the model is SWA / paged swa pool.)
            self.cache_manager.maybe_free_swa_out_of_window(
                batch.reqs, forward_iter=self._forward_iter)
            for req in batch.reqs:
                req.decode_batch_idx += 1
        else:
            # Prefill sibling of the decode driver: free out-of-window swa BEFORE allocating
            # this chunk, so a chunked prompt longer than the swa pool never accumulates its
            # whole swa footprint (which would exhaust alloc_swa). No-op unless SWA/paged.
            self.cache_manager.free_swa_out_of_window_extend(batch.reqs)
        # before the widening below, so the penalty check reads each request's real remain_len
        sample_args = self.engine.sampler.prepare(batch)
        # widened after the window free above, whose threshold reads device_len
        if batch.phase == "verify":
            for req in batch.reqs:
                req.widen_pending(batch.rows_per_req - 1)
        # Polymorphic page allocation: DSV4 allocates window pages + cmp/idx blocks into its
        # slot maps; the generic manager allocates KV pages into the page table.
        self.cache_manager.allocate_paged(batch.reqs)
        if batch.phase == "verify":
            # the drafts run on the GPU while the host builds this batch's metadata below
            self.spec.propose_into(batch.reqs, self.token_pool)
        if batch.is_prefill:
            self._gather_multimodal(batch)
        batch.positions = _make_positions(batch, self.device)
        if self._model_is_mrope:
            batch.mrope_positions = _make_mrope_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = None if batch.phase == "verify" else _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        if self.engine.linear_backend is not None:
            self.engine.linear_backend.prepare_metadata(batch)
        if not batch.is_prefill:
            # This batch's padded per-row page-table rows. Backends that snapshot the table for
            # a captured replay (DSV4) read them in prepare_metadata / prepare_for_replay.
            batch.active_table_idx = input_mapping[0][:: batch.rows_per_req].view(-1)
        self.engine.attn_backend.prepare_metadata(batch)
        if (
            self.layer_major
            and batch.is_prefill
            and len(batch.reqs) == 1
            and batch.reqs[0].extend_len > self.prefill_budget
            and batch.reqs[0].extend_len <= _LAYER_MAJOR_MAX_EXTEND_TOKENS
            and getattr(batch.reqs[0], "mm_items", None) is None
        ):
            batch.layer_major = True
        return ForwardInput(
            batch=batch,
            sample_args=sample_args,
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _gather_multimodal(self, batch: Batch) -> None:
        """Plan the chunk's encoder jobs, gather rows and scatter rows over the batch; the engine runs them before the LM forward."""
        jobs, plan, rows, block_ends = plan_mm_batch(batch.padded_reqs, self.engine.encoder_cache)
        if plan:
            batch.mm_encoder_jobs = jobs
            batch.mm_gather_plan = plan
            batch.mm_rows = torch.tensor(rows, dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
            batch.mm_block_ends = torch.tensor(block_ends, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        if self._bidirectional_mm and not self._warned_cut_image and (cut := cut_image_spans(batch.padded_reqs)):
            # only a bidirectional image span loses context when cut, and only an image longer than the chunk still gets cut
            lo, hi = cut[0]
            self._warned_cut_image = True
            logger.warning_rank0(
                f"an image of {hi - lo} tokens does not fit one prefill chunk (--max-extend-tokens {self.prefill_budget}, or the sliding-window pool's share of it): its earlier rows attend within the first part only"
            )

    def _schedule_next_batch(self) -> ForwardInput | None:
        # Prefill keeps priority -- a request that never prefills never starts -- but a
        # long run of prefill chunks must not starve in-flight decodes. When the policy
        # says a decode is due, take one if a decode is actually runnable; otherwise stay
        # with prefill (never spend a slot idle just to keep a promise).
        # getattr, not attribute access: the accounting tests drive this method on a
        # partially built Scheduler, and interleaving must be an opt-in that a stripped
        # object simply does not have rather than something that breaks it. A missing
        # policy is the historical prefill-first order.
        policy = getattr(self, "_interleave", None)
        batch = None
        if policy is not None and policy.wants_decode():
            batch = self.decode_manager.schedule_next_batch()
            if batch is not None:
                policy.note_decode()
        if batch is None:
            batch = self.prefill_manager.schedule_next_batch(self.prefill_budget)
            if batch is not None and policy is not None:
                policy.note_prefill()
        if batch is None:
            batch = self.decode_manager.schedule_next_batch()
            if batch is not None and policy is not None:
                policy.note_decode()
        if batch is None:
            return None
        forward_input = self._prepare_batch(batch)
        self._report_prompt_admissions(batch)
        return forward_input

    def _report_prompt_admissions(self, batch: Batch) -> None:
        """Publish first-prefill accounting only after batch preparation succeeded.

        ``send_result`` is rank-aware: TP rank 0 forwards the signal, other ranks are
        no-ops. The offline handler explicitly ignores this online-accounting message.
        """
        if not batch.is_prefill or not batch.prompt_admissions:
            return
        self.send_result(
            [
                PromptAdmittedMsg(uid=uid, prompt_tokens=prompt_tokens, cached_tokens=cached_tokens)
                for uid, prompt_tokens, cached_tokens in batch.prompt_admissions
            ]
        )

    def _flush_abort_acks(self) -> None:
        pending = getattr(self, "_pending_abort_acks", None)
        if not pending:
            return
        uids = sorted(pending)
        pending.clear()
        self.send_result([ErrorReplyMsg(uid=uid, error="request aborted") for uid in uids])

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        if self.toolcall_anchor_id is not None and not batch.is_prefill:
            self.cache_manager.snapshot_toolcall_anchor(batch.reqs)
        if batch.phase == "verify":
            return self.spec.verify_step(batch, sample_args)
        forward_output = self.engine.forward_batch(batch, sample_args)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output


def _make_mrope_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    """[3, N] rope rows: an image request's prompt tokens use their precomputed columns, everything else is sequence index + per-request delta."""
    needed = sum(r.extend_len for r in batch.padded_reqs)
    host = torch.empty((3, needed), dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        out = host[:, offset : offset + length]
        full = req.mrope_positions_full
        if full is not None and req.device_len <= full.shape[1]:
            out.copy_(full[:, req.cached_len : req.device_len])
        else:
            row = torch.arange(
                req.cached_len + req.mrope_delta,
                req.device_len + req.mrope_delta,
                dtype=torch.int32,
            )
            out.copy_(row.unsqueeze(0).expand(3, -1))
        offset += length
    return host.to(device, non_blocking=True)


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
