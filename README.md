<div align="center">
  <img alt="FreeToken - pretor's fork (4x5060ti qwen3.8 FN)" src="assets/freetoken-pretor-fork.png" width="70%">
</div>

<p align="center">
| <a href="https://www.flashml.ai/"><b>Download</b></a> | <a href="https://arxiv.org/abs/2608.16157"><b>Paper</b></a> | <a href="https://join.slack.com/t/flashml/shared_invite/zt-3zpdh5j10-9dwTXrgLiqpVxizhA9KVbA"><b>Developer Slack</b></a> | <a href="https://discord.gg/MsA277cJzZ"><b>Community Discord</b></a> | <a href="https://github.com/FlashML-org/FreeToken/issues/482"><b>Community WeChat</b></a> |
</p>


Unlock datacenter-class intelligence on the hardware you already own — Run 290B+ frontier MoE models locally on your gaming PC at blistering interactive speeds.

## This Fork: 4x RTX 5060 Ti & Consumer Multi-GPU Optimization

This fork by **[@pretor](https://github.com/pretor)** is specifically engineered and battle-tested for high-throughput serving of frontier MoE architectures on consumer multi-GPU workstations (specifically **4x NVIDIA GeForce RTX 5060 Ti 16GB** over PCIe 3.0, without NVLink or P2P).

### What This Fork Adds (Beyond Upstream `main`):

1. **Opt-In Layer-Major Long Prefill for Multi-GPU TP=4 (`--prefill-layer-major`)**:
   - **Upstream Adaptation**: Adapts upstream PR #686 (which only targeted single-GPU TP=1) for multi-GPU TP=4 and resolves multi-chunk activation memory management.
   - **The Problem It Solves**: Standard chunked prefill divides long contexts into chunks of `--max-extend-length 4096`. For a 32k prompt, 8 chunks traverse all layers sequentially, requiring MoE experts to be re-streamed across PCIe 8 separate times per layer.
   - **How Layer-Major Works**: Inverts execution order so each decoder layer processes all chunks consecutively while its MoE experts remain resident in GPU VRAM. Experts traverse PCIe only once per layer instead of once per chunk, nearly doubling prefill throughput.
   - **Zero-Copy VRAM Activation Flow**: Retains chunk intermediate activations directly in VRAM without costly CPU roundtrips, paired with immediate chunk lifecycle cleanup (`hidden.clear()`) and post-prefill cache release (`torch.cuda.empty_cache()`).
   - **Sampling Resilience on Primary GPUs**: Added automatic fallback to Triton sampling if FlashInfer's raw `cudaMalloc` encounters driver memory fragmentation during non-zero temperature sampling on GPU 0 (which often drives desktop displays).
   - **Measured Performance on 4x RTX 5060 Ti 16GB**:
     - **16k Prefill**: **2,082 tok/s** (TTFT: 7.90s, Decode: 63.5 tok/s)
     - **32k Prefill**: **2,072 tok/s** (TTFT: 15.86s, Decode: 58.7 tok/s)
     - **64k Prefill**: **1,708 tok/s** (TTFT: 38.44s, Decode: 63.1 tok/s)
     - **110k Prefill**: **1,600 tok/s** (TTFT: 68.80s, Decode: 60.0 tok/s)
     - **198k Prefill (99% full 200k KV limit)**: **1,475 tok/s** (TTFT: 134.29s, Decode: 60.2 tok/s)

2. **MXFP8 Tensor-Parallel (TP) Weight Sharding (by @pretor)**:
   - **Custom Implementation**: Created and integrated full Tensor-Parallel (TP4) row/column weight sharding for **MXFP8** dense layers, merger projections, and shared experts.
   - **Why It Matters**: Upstream FreeToken only supported single-GPU (TP=1) or duplicated layouts for MXFP8 weights. This sharding allows modern hybrid checkpoints—which pair NVFP4 routed experts with MXFP8 shared experts—to partition weights cleanly across 4 GPUs, saving precious VRAM and unlocking true multi-GPU scaling.

3. **Vision Tower TP Sharding (Multi-Modal Acceleration)**:
   - Shards vision encoder weights across all TP ranks (`_shard_vision_tensor`), dividing ViT memory to only **~214 MB per GPU** instead of duplicating 856 MB on every card or crashing under TP > 1.

4. **CPU Vision Encoder: the Vision Tower in RAM, Zero VRAM (`--mm-encoder-weights cpu`)**:
   - **What**: image input on Qwen3.8-Flash-Next without spending VRAM. The vision tower (27-block ViT, ~449M params) lives once in host RAM in FP32 and runs in its own CPU process (`freetoken-vision-cpu`, no CUDA device visible), modelled on [strata-nvfp4](https://github.com/sergqwer/strata-nvfp4)'s `strata-vision` sidecar.
   - **How it works**: the tokenizer sends only the requests that carry images to the encoder process. It replaces each image's pixels with the tower's output rows and passes the request on to the scheduler. The 4 TP ranks build and load no tower, so the KV cache and the expert cache keep their full size, and running requests keep decoding while an image encodes. Text-only requests never pass through it.
   - **Why FP32 on the CPU**: these Xeons have no BF16 or AMX units. BF16 was ~2x slower than FP32 and ~11-12% off it at 1024 tokens, the same drift the BF16 GPU tower shows against FP32.
   - **Knobs**:
     - Images are capped at 1024 tokens unless `--image-max-tokens` is given.
     - `--mm-encoder-cache-mb` (default 256) keeps recently encoded images, so an image a chat client resends every turn is encoded once.
     - `--mm-encoder-threads` / `--mm-encoder-cpus` control placement. By default it uses one NUMA node's physical cores, not pinned; that measured fastest.
   - **Measured on this ThinkStation (2x Xeon Gold 6152, 22 threads)**:
     - Encode time per image: 1.1 s at 256 tokens, 2.0 s at 484, 5.4 s at 1024, 16.6 s at 2025.
     - Against `--text-model-only`: VRAM grows by ~20 MiB per GPU, the KV cache (200k tokens) and expert cache are unchanged, text decode is unchanged (~62-64 tok/s), and the encoder process takes ~2.4 GB RSS.
     - End to end: the model reads image content correctly (colours, text in the picture), a resent image comes from the cache, a corrupt image gets a clean 400, and an abort mid-encode is handled.
   - Details: [CPU vision encoder](docs/cli.md#cpu-vision-encoder); benchmark: `benchmarks/bench_cpu_vision.py`.

5. **Pre-Merged Cutting-Edge Upstream PRs**:
   - **[PR #735](https://github.com/FlashML-org/FreeToken/pull/735)**: `perf(moe): pipeline the nvfp4 prefill scale loads on geforce sm_86 / sm_89` — pipelines e4m3 block scale loads via `cp.async` (as 32-bit words), with our custom extension enabling SM_120 (`(12, 0)`) with 8 warps.
   - **[PR #736](https://github.com/FlashML-org/FreeToken/pull/736)**: `perf(moe): prefill nvfp4 moe experts on nvfp4 activations on sm_120` — W4A4 MoE prefill Gluon kernels on SM_120 for checkpoints with static activation scales.
   - **[PR #733](https://github.com/FlashML-org/FreeToken/pull/733)**: `perf(moe): speed up the nvfp4 moe prefill gemm on rtx gpus` — warp-specialized Gluon NVFP4 prefill GEMM for SM_120 with TMA and `mma.sync`.
   - **[PR #734](https://github.com/FlashML-org/FreeToken/pull/734)**: `feat(moe): bench hybrid fetch fractions per moe layer kind` — per-layer-kind bandwidth benchmarking for hybrid offload.
   - **[PR #729](https://github.com/FlashML-org/FreeToken/pull/729)**: `perf(kernel): split triton decode attention by context length and sm count` — splits decode attention across SMs for higher decode throughput.
   - **[PR #726](https://github.com/FlashML-org/FreeToken/pull/726)**: `perf(attention): copy triton metadata without blocking the host` — asynchronous non-blocking Triton metadata transfer.
   - **[PR #723](https://github.com/FlashML-org/FreeToken/pull/723)**: `perf(kernel): rewrite the triton nvfp4 moe prefill gemm` — rewritten Triton NVFP4 MoE prefill GEMM optimized for GeForce RTX GPUs.
   - **[PR #635](https://github.com/FlashML-org/FreeToken/pull/635)**: `perf(moe): avoid intra-op fanout while filling NVFP4 host banks` — eliminates intra-op fanout latency when populating host-side NVFP4 expert banks.
   - **[PR #636](https://github.com/FlashML-org/FreeToken/pull/636)**: `feat(quant): opt-in fp8-block QAT for the lm_head` — enables FP8-block quantized LM heads to conserve GPU memory.
   - **[PR #639](https://github.com/FlashML-org/FreeToken/pull/639)**: `feat(server): add --api-key authentication` — adds `--api-key` and `$FREETOKEN_API_KEY` bearer authentication to protect the server endpoint.

6. **Production Stability & 200k Context Resilience**:
   - Custom timeout flags (`--step-timeout 120`, `--distributed-timeout 2592000`) that prevent NCCL heartbeat and rendezvous timeouts during long idle stretches or heavy prefill batches.
   - Production-verified **FP8 KV-Cache** across up to **200,000 tokens** (~1.31 GiB VRAM per GPU for 200k tokens).

7. **Prefill PCIe overlap (`FREETOKEN_PREFILL_COPY_DEFER`, `FREETOKEN_MOE_AR_GROUP`)**:
   - **The link**: on the ThinkStation P920 the four 5060 Ti cards have no P2P. The TP all-reduce of the bf16 hidden state `[T, 2560]` and the expert H2D copies share the host and PCIe 3.0 (the slots are x16-capable and often come up at x8 while idle). A microbench of a 21 MiB all-reduce against a 150 MiB H2D slowed both at once: all-reduce **+39%**, H2D **+35%**. On a 16k prefill the all-reduce is about half the chunk, the expert copies take about **2.1 s**, and the compute stream waits on those copies for only **~50 ms**. The copies are already mostly hidden. The all-reduce is the exposed PCIe cost.
   - **`FREETOKEN_PREFILL_COPY_DEFER=prev_ar`** (default `off`): double-buffered prefetch of the next layer's experts used to start when the current layer's GEMMs finished, so that H2D sat on the wire together with the MoE all-reduce. `prev_ar` records the buffer-release event in `mark_layer_reduced`, after the all-reduce is queued, and the next copy starts once the reduce has the link. `off` keeps the copy at release time.
   - **`FREETOKEN_MOE_AR_GROUP`** (default `0`, one all-reduce for the whole chunk; production uses `2048`): the capy-flash T62 overlap, with the payload left in bf16. A prefill chunk is split into token groups, so a 4096-token extend is two groups. Each group's routed GEMM stays on the compute stream and that group's all-reduce runs on a side stream. PyNCCL is aimed at that stream with `tvm_ffi.use_torch_stream`, because the kernel launches on `TVMFFIEnvGetStream`. The next group's GEMM therefore covers the previous group's all-reduce. The last group of a layer stays exposed: the next layer needs the reduced hidden state. `mark_layer_reduced` runs after the compute stream has waited for every group, so `prev_ar` still starts the next copy after this layer's all-reduces.
   - **Measured** on the production shape (layer-major prefill, hit-D2D, `--moe-cache-size 14500`, demand-prefill 512, FP8 KV reserved at 200k, `--max-extend-length 4096`), Qwen3.8-Flash-Next NVFP4, TP=4. `prev_ar` alone, against `off`: **64k prefill 1687 → 1795 tok/s (+6.4%)**; 4k and 16k flat; greedy text identical. Grouped all-reduce `2048` on top of `prev_ar`, three reps (the first 4k rep hit the radix cache and is left out of these means):

     | prompt | `FREETOKEN_MOE_AR_GROUP=0` | `2048` |
     | --- | --- | --- |
     | 4k | 1621 tok/s | 1635 tok/s (+0.8%) |
     | 16k | 2057 tok/s | 2164 tok/s (+5.2%) |
     | 64k | 1797 tok/s | 1852 tok/s (+3.1%) |

     With both switches on, 64k prefill moves from **1687 tok/s** before either change to **1852 tok/s**. The first generated token stayed `We`. Greedy continuations on the grouped path stayed close to the ungrouped run; overlapping the all-reduce onto another stream changes reduction order, so the text can differ.
   - **Measured and left out of the tree**: capy-flash cache policies (`lru_admit`, a frequency admission, and `fr`) on this per-layer cache (~302 of 512 experts resident, hit-D2D **54.8%**). Prefill and decode throughput did not move. Decode stayed ~60 tok/s with about 10–14 expert misses per token. A literal capy-style `ar` defer, recording the copy release at a different point than `prev_ar`, was slower than `prev_ar`.
   - **Tests**: `tests/moe/test_prefill_copy_defer.py`, `tests/moe/test_moe_ar_group.py`. Both paths stay off unless the env vars are set.

8. **Elastic KV (`--kv-initial-tokens`, `--kv-max-tokens`)**:
   - **What**: the KV pool starts small so the GPU expert cache can be large, then grows up to a configured ceiling when a request needs more context. Experts stay in host RAM (`--moe-strategy offload`). Growth only changes how many of them stay resident on the GPU. The scheduler does this while idle, on every TP rank, with more than one running request allowed. A request that fits the current pool is admitted immediately.
   - **How to turn it on**: set both flags. `--kv-initial-tokens` is the pool occupied at startup. `--kv-max-tokens` is the ceiling (capped by the checkpoint context and `--max-seq-len-override`, so raise the override to the same value). `--moe-cache-size` is the startup slot pin. Omit either KV flag and the server keeps the fixed pool.
   - **What a grow costs**: extra KV pages, plus a fixed 384 MiB scratch so the layer-major tail of a long prefill still fits. The scratch does not grow with the prompt. CUDA graphs are recaptured once per grow.
   - **Measured on this ThinkStation** (Qwen3.8-Flash-Next NVFP4, TP=4, FP8 KV, layer-major, `prev_ar`, `FREETOKEN_MOE_AR_GROUP=2048`). 16450 slots is the largest cache that finishes a 32k prefill; 16550 OOMs. With `--kv-initial-tokens 40960` and `--kv-max-tokens 262144`:

     | prompt | GPU expert slots | prefill |
     | --- | --- | --- |
     | 16k | 16450 | 1999 tok/s |
     | 32k | 16450 | 2222 tok/s |
     | 64k | 15781 | 1372 tok/s |
     | 128k | 15121 | 1201 tok/s |
     | 160k | 14791 | 1253 tok/s |
     | 200k | 14378 | 1263 tok/s |
     | 262k (model max) | 13802 | 1248 tok/s |

   - **Tests**: `tests/engine/test_elastic_kv.py`.

9. **Blackwell SM_120 Gluon Kernels & W4A4 MoE Acceleration (PR #733 & PR #736)**:
   - **Upstream PR #733 & PR #736 Integration**: Merged warp-specialized **Gluon NVFP4 MoE kernels** specifically targeting Blackwell SM_120 (GeForce RTX 50 series), alongside PR #736's **W4A4 MoE prefill** path.
   - **Multi-GPU TP Sharding Integration**: Adapted PR #736's static activation scale (`_input`) ingestion in `nvfp4_banks.py` to seamlessly operate alongside Tensor Parallelism (`tp_size=4`), correctly bypassing intermediate dimension slicing on activation scale scalars.
   - **Split Decode Attention (PR #729)**: Integrated context-length and SM-count aware attention splitting across SMs.
   - **Robust ABI Import Safety (`freetoken/kernel/backend.py`)**: Enhanced `_importable` to dynamically verify C/CUDA ABI compatibility (e.g. `sgl_kernel` / `torch` 2.14.x) preventing silent runtime symbol crashes and seamlessly falling back to Triton.
   - **Production Benchmark Results on 4x RTX 5060 Ti TP=4 (Qwen3.8-Flash-Next-NVFP4-QAD)**:
     - **50 Tokens Prefill / 50 Decode**: TTFT **0.222 s (222 ms)**, Decode **70.15 tok/s**, Total: **0.92 s**
     - **1k Prefill / 200 Decode**: TTFT **1.767 s**, Decode **67.67 tok/s**, Total: **4.71 s**
     - **4k Prefill / 200 Decode**: TTFT **2.452 s**, Prefill: **1,707.16 tok/s**, Decode: **65.15 tok/s**, Total: **5.51 s**
     - **8k Prefill / 200 Decode**: TTFT **3.889 s**, Prefill: **2,130.10 tok/s**, Decode: **64.10 tok/s**, Total: **6.99 s**
     - **16k Prefill / 500 Decode (3x Verified Avg)**: TTFT **7.20 s**, Prefill: **2,287.34 tok/s**, Decode: **65.26 tok/s**, Total: **14.85 s**
     - **32k Prefill / 2,000 Decode**: TTFT **18.30 s**, Prefill: **1,797.25 tok/s**, Decode **64.23 tok/s** constant over 2k tokens, Total: **49.42 s**
     - **64k Prefill / 500 Decode**: TTFT **36.73 s**, Prefill: **1,788.14 tok/s**, Decode: **64.16 tok/s**, Total: **44.50 s**

10. **Pipelined NVFP4 Scale Loads (PR #735) & Extended Context Scaling (up to 150k)**:
   - **Upstream PR #735 Integration**: Pipelines e4m3 scale reading into int32 words through `cp.async` for Triton inline-dequant MoE prefill GEMM.
   - **Blackwell SM_120 Tuning**: Extended tuning configuration (`_PREFILL_TUNING`) to explicitly activate `scale_words` and 8 warps on `(12, 0)` Blackwell GeForce GPUs.
   - **Production Headroom & MoE Slot Tuning**: Tuned base MoE slot allocation to 16,350 slots, guaranteeing stable memory headroom for massive context extensions without triggering CUDA Out-of-Memory.
   - **Measured Production Benchmarks on 4x RTX 5060 Ti TP=4 (Qwen3.8-Flash-Next-NVFP4-QAD)**:
     - **1k Prefill / 100 Decode (3x Verified Avg)**: TTFT **1.77 s**, Prefill: **627.08 tok/s** (+16.1% vs baseline), Decode: **64.90 tok/s**, Total: **3.29 s**
     - **8k Prefill / 1,000 Decode (3x Verified Avg)**: TTFT **3.91 s**, Prefill: **2,114.53 tok/s**, Decode: **67.44 tok/s**, Total: **18.73 s**
     - **16k Prefill / 500 Decode (3x Verified Avg)**: TTFT **7.11 s**, Prefill: **2,316.11 tok/s** (+1.3% over PR #736), Decode: **67.46 tok/s**, Total: **14.51 s**
     - **32k Prefill / 1,000 Decode**: TTFT **13.06 s**, Prefill: **2,445.83 tok/s**, Decode: **67.87 tok/s**, Total: **27.78 s**
     - **64k Prefill / 1,000 Decode**: TTFT **52.85 s**, Prefill: **1,200.11 tok/s**, Decode: **58.25 tok/s**, Total: **70.00 s**
     - **150k Prefill / 1,000 Decode**: TTFT **125.14 s**, Prefill: **1,199.48 tok/s**, Decode: **58.76 tok/s**, Peak VRAM: **16,189.4 MiB** (121.6 MiB headroom preserved, zero OOM), Total: **142.14 s** (2 min 22 s)

11. **Production Offload Transport Overhauls: Priority Copy Stream, CUDA-Graph Shared-Expert Overlap & TMA Hardware Gather (Issues #752, #753, #751)**:
   - **[Issue #752](https://github.com/FlashML-org/FreeToken/issues/752) — Prefill Copy Stream Priority & Epoch Invalidation (`FREETOKEN_PREFILL_COPY_PRIORITY`, `FREETOKEN_PREFILL_EPOCH_INVALIDATE`)**:
     - **The Bottleneck**: Upstream ran prefill double-buffer copy streams at default priority (`0`), launching ~240 small bookkeeping kernels (`_invalidate_prefill_buffer`) per prefill directly on the copy stream. Those kernels sat in queue behind the compute GEMMs, stalling the hardware DMA engine for ~1.4 ms per layer.
     - **The Fix**: Prefill copy stream is elevated to `priority=-1` (highest hardware dispatch priority), and both double buffers are unmapped once per prefill epoch in `begin_prefill()` on the compute stream. The copy stream now carries exclusively DMA work (`cudaMemcpyBatchAsync`), eliminating all bookkeeping bubbles.
   - **[Issue #753](https://github.com/FlashML-org/FreeToken/issues/753) — Decode Miss Gather & Shared-Expert Overlap in CUDA Graph (`FREETOKEN_DECODE_SHARED_OVERLAP`)**:
     - **The Bottleneck**: In offloaded decode, each layer executed `ensure_experts`, `copy_missing` (PCIe gather of un-cached routed experts), shared expert computation, and routed GEMM sequentially on the compute stream—even though the shared expert does not depend on missed routed experts.
     - **The Fix**: Introduced `DecodeSharedOverlap` with pre-materialized CUDA event pairs (`ready` and `done`) for all 40 layers. The PCIe miss gather (`copy_missing`) forks to a dedicated background stream while the shared expert MLP (`gate_up_proj` + `silu_and_mul` + `down_proj`) computes concurrently on the compute stream. They join seamlessly before `_expert_gemm` and are 100% captured inside the decode CUDA graph across TP=4 ranks without any host synchronization.
   - **[Issue #751](https://github.com/FlashML-org/FreeToken/issues/751) — TMA Asynchronous Bulk Transfer Gather (`FREETOKEN_DECODE_GATHER=tuned`)**:
     - **The Bottleneck**: Legacy decode expert gather dispatched `fast_index_copy_multi` across 48 CTAs x 1024 threads issuing small 16-byte `ld.global` instructions, causing memory queue pressure across PCIe.
     - **The Fix**: Implemented hardware Tensor Memory Accelerator (TMA) bulk transfers via `DecodeGatherTmaKernel` (`decode_gather.cuh`). Total missed expert bytes are split evenly across 8 CTAs using a 3-stage shared-memory ring (16 KiB chunks, 48 KiB shared memory).
     - **Blackwell SM_120 Bug Prevention**: Standard `.shared::cluster` TMA triggers an NVIDIA driver bug on SM_120 that inflates per-thread stack budget to 14,416 B, leaking 1.37 GiB VRAM per GPU. Our implementation strictly targets native `.shared::cta` (`cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes`), ensuring **0 B VRAM overhead** and 100% bit-exact expert gathers.
   - **Measured Production Benchmarks on 4x RTX 5060 Ti TP=4 (Qwen3.8-Flash-Next-NVFP4-QAD)**:
     - **1k Prefill / 1,000 Decode**: TTFT **1.81 s**, Prefill: **611.16 tok/s**, Decode: **67.84 tok/s**, Total: **16.54 s**
     - **8k Prefill / 1,000 Decode**: TTFT **3.87 s** (down from 3.97 s baseline), Prefill: **2,137.56 tok/s**, Decode: **68.11 tok/s**, Total: **18.58 s**
     - **16k Prefill / 500 Decode (3x Verified Avg)**: TTFT **7.24 s**, Prefill: **2,274.03 tok/s**, Decode: **68.66 tok/s** (Peak: **68.99 tok/s**, +4.6% vs baseline), Total: **14.51 s**
     - **16k Prefill / 1,000 Decode (3x Verified Avg)**: TTFT **7.21 s** (steady across all runs), Prefill: **2,282.63 tok/s**, Decode: **67.90 tok/s**, Total: **21.93 s**
     - **32k Prefill / 1,000 Decode**: TTFT **12.88 s**, Prefill: **2,481.53 tok/s**, Decode: **66.83 tok/s**, Total: **27.82 s**
     - **Concurrent Throughput (2x 16k Prefill + 1,000 Decode)**: Processed **34,143 tokens** in **30.70 s** — Aggregate Prefill: **2,449.67 tok/s**, Aggregate Decode: **113.62 tok/s**, System Throughput: **1,112.01 tok/s**.

12. **Upstream Synchronization (`upstream/main` @ `cc56b6b`) & PR #742 (`feat(moe): fetch locked-layer experts through pinned staging`)**:
   - **Upstream Sync (`cc56b6b`)**: Merged all upstream updates including:
     - **[PR #759](https://github.com/FlashML-org/FreeToken/pull/759)**: Strict weight reading adhering to checkpoint-declared `QuantConfig` (`_DenseReader`).
     - **[PR #760](https://github.com/FlashML-org/FreeToken/pull/760)**: GGUF checkpoint loading with quantized indexers for `qwen4_exp`.
     - **[PR #755](https://github.com/FlashML-org/FreeToken/pull/755)**: Runtime zero-centered RMSNorm (+1 in FP32) stabilization.
     - **[PR #721](https://github.com/FlashML-org/FreeToken/pull/721)**: GLM-5.3-Flash GGUF checkpoint serving support on `glm5-next`.
     - **[PR #762](https://github.com/FlashML-org/FreeToken/pull/762) & [PR #756](https://github.com/FlashML-org/FreeToken/pull/756)**: Checkpoint diagnosis, tensor renaming, and GGUF inspection tools (`scripts/checkpoint_diagnose/`, `scripts/gguf_check/`).
   - **PR #742 Integration**: Incorporated Xiaoze Fan's locked-layer expert fetch pipeline:
     - Dedicated multi-threaded pinned staging buffer (`_cpu_moe.LockedFetch`, `python/freetoken/moe/locked_fetch.py`).
     - Accelerated gathering of locked-layer expert misses via `fast_index_copy_multi` in C++/CUDA (`locked_fetch.cpp`, `locked_fetch_io.cu`).
     - Recompiled and linked C++/CUDA extensions with native Blackwell SM_120 support.
   - **Full Backward Compatibility & TP=4 Preserved**:
     - Maintained automatic on-the-fly FP16 -> FP32 NVFP4 global scale upcasting for legacy FTW checkpoints.
     - Retained complete multi-GPU TP=4 tensor sharding and `/dev/shm` shared memory PLE table mapping.

---

### Tested & Verified Models:

These models have been tested and run at full speed on this fork:

1. **[local-inference-lab/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4)**
   - *Architecture*: 177B total parameter MoE, Quantization-Aware Distilled (QAD).
   - *Quantization*: NVFP4 routed experts + MXFP8 shared experts, BF16 attention & embeddings.
   - *Performance*: Achieves **~62–65 tokens/sec decode** on 4x RTX 5060 Ti with interactive latency.
   - *Vision*: image input served by the CPU vision encoder (`--mm-encoder-weights cpu`), with no VRAM spent on the vision tower.

2. **[RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)** (Base: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next))
   - *Architecture*: ModelOpt-quantized candidate release of Qwen3.8 Flash Next.
   - *Quantization*: W4A4 NVFP4 routed experts with standard dense layers.

---

### Production Launch Setup (4x RTX 5060 Ti TP=4)

Below is our production launch configuration used on Lenovo ThinkStation P920 (4x RTX 5060 Ti 16GB):

```bash
# 1. Convert checkpoint to FTW format (pre-seeds expert banks for instant load)
ft checkpoint convert \
  --model local-inference-lab/Qwen3.8-Flash-Next-NVFP4 \
  --output /models/local-inference-lab-Qwen3.8-Flash-Next-NVFP4-FTW

# 2. Launch FreeToken server (TP=4, FP8 KV, elastic 41k→262k, 16,450 MoE slots, layer-major, image input on CPU)
#    prev_ar keeps expert H2D off the wire until the MoE all-reduce is queued.
#    MOE_AR_GROUP=2048 overlaps that all-reduce with the next token group's GEMM.
#    kv-initial-tokens is the pool at startup; kv-max-tokens is the ceiling the pool may grow to.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FREETOKEN_PREFILL_COPY_DEFER=prev_ar
export FREETOKEN_MOE_AR_GROUP=2048

numactl --interleave=all ft serve \
  --model /models/local-inference-lab-Qwen3.8-Flash-Next-NVFP4-FTW \
  --tp-size 4 \
  --gpu 0,1,2,3 \
  --moe-strategy offload \
  --quant-backend moe.nvfp4=triton \
  --ple-backend pinned \
  --expert-load serial \
  --embed-device cpu \
  --decode-interleave-every 4 \
  --moe-cache-size 16450 \
  --memory-ratio 0.95 \
  --moe-cache-scope per-layer \
  --moe-prefill-hit-d2d \
  --prefill-layer-major \
  --moe-prefill-demand-tokens 512 \
  --kv-cache-dtype fp8 \
  --kv-initial-tokens 40960 \
  --kv-max-tokens 262144 \
  --max-seq-len-override 262144 \
  --max-running-requests 2 \
  --cuda-graph-max-bs 2 \
  --max-extend-length 4096 \
  --mamba-host-slots 32 \
  --served-model-name Qwen3.8-Flash-Next-NVFP4-QAD \
  --mm-encoder-weights cpu \
  --step-timeout 120 \
  --distributed-timeout 2592000 \
  --host 0.0.0.0 --port 8000

# 3. Send an image (OpenAI API; an http(s) URL or a base64 data: URL)
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next-NVFP4-QAD",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "https://example.com/picture.png"}},
    {"type": "text", "text": "What is in this picture?"}]}]}'
```

> [!TIP]
> **Key Performance Settings for Long Contexts & Stability:**
> - `--prefill-layer-major`: Reorders prefill so all chunks run through a layer before advancing, keeping MoE experts resident and accelerating long-context TTFT from ~1,400 tok/s up to **2,082 tok/s**.
> - `--kv-initial-tokens 40960`, `--kv-max-tokens 262144`, and `--moe-cache-size 16450`: start with a 41k KV pool and 16450 GPU expert slots, and grow the pool up to the 262k model maximum. On this box that is the largest slot count that still finishes a 32k prefill; at 200k the cache settles near **14378** slots.
> - `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: Eliminates memory fragmentation spikes during long multi-chunk requests.
> - `FREETOKEN_PREFILL_COPY_DEFER=prev_ar` and `FREETOKEN_MOE_AR_GROUP=2048`: keep expert H2D off the PCIe link while the MoE all-reduce runs, then hide each group's all-reduce under the next group's GEMM. On this box that is **64k prefill 1687 → 1852 tok/s** before the elastic pool. Unset, both default off (`off` and `0`).

`--mm-encoder-weights cpu` serves images with the vision tower on the CPU (see item 4 above); images are
capped at 1024 tokens unless `--image-max-tokens` is given. For text-only serving, use `--text-model-only`
instead, which also saves the encoder process's ~2.4 GB of RAM.

---

## About

FreeToken is an edge-native Mixture-of-Experts (MoE) serving engine designed for running frontier-scale open-weight models on personal and consumer hardware. It treats heterogeneous edge resources—GPUs, CPUs, host memory, and interconnects—as a unified, elastic inference platform. Its core features include:  

- **Fast Edge-Native Runtime**: Provides efficient MoE serving with bandwidth-adaptive CPU–GPU co-execution ($q^\star$ policy), full-layer double-buffered prefill streaming, global LRU expert caching, graph-compatible execution, and the FTW fast weight format.  
- **Semantic-Aware Caching**: Features semantic anchor checkpoints for recurrent state and KV caches, allowing agentic context edits (e.g., tool calls, thinking blocks) to avoid redundant context recomputation.  
- **Elastic Memory Management**: Supports dynamic, runtime VRAM re-allocation between expert caches and KV memory without engine restarts or weight reloading.  
- **Broad MoE & Ecosystem Support**: Supports frontier open-weight MoE models (e.g., DeepSeek-V4-Flash, Qwen3.6-35B-A3B, GLM-5.2) across various parameter scales and quantization formats (e.g., MXFP4, NVFP4, FP8, BF16, GGUF), with Anthropic/OpenAI-compatible APIs for seamless integration with real-world coding and tool-calling agents (e.g., Codex, Claude Code, OpenCode, OpenClaw, DeepSeek Harness). 
- **Diverse Consumer Hardware**: Scales across consumer laptops, gaming desktops, and workstation GPUs, with native support for NVIDIA RTX 30, RTX 40, and RTX 50 series GPUs.  

## Getting Started

### Desktop app

Download FreeToken for Windows or Linux at [flashml.ai](https://www.flashml.ai/). It sets the engine up for you and gives you a GUI for running models, chatting, and tuning the engine.

<div align="center">
  <img alt="FreeToken Desktop" src="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/desktop-console.png" width=92%>
</div>

### CLI

Install FreeToken with [uv](https://docs.astral.sh/uv/) (recommended) or pip:

```bash
uv pip install "freetoken[accel]"
```

Or build from source:

```bash
git clone https://github.com/FlashML-org/FreeToken.git && cd FreeToken
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```

For More details:

- [Install FreeToken](https://github.com/FlashML-org/FreeToken/blob/main/docs/install.md)
- [Quick start](https://github.com/FlashML-org/FreeToken/blob/main/docs/quickstart.md)
- [Supported models](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md)
- [CLI reference](https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md)
- [Repairing old FTW checkpoints](https://github.com/FlashML-org/FreeToken/blob/main/docs/ftw-hotfix.md)
- [Diagnosing a checkpoint that fails to load](https://github.com/FlashML-org/FreeToken/blob/main/scripts/checkpoint_diagnose/README.md)

## Citation

If you use FreeToken for your research, please cite our [paper](https://arxiv.org/abs/2608.16157):

```bibtex
@article{yang2026freetoken,
  title={FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution},
  author={Yang, Shuo and Fan, Xiaoze and Pan, Melissa and Xi, Haocheng and Wang, Zhe and Sun, Shanlin and Keutzer, Kurt and Han, Song and Zaharia, Matei and Xu, Chenfeng and Stoica, Ion},
  journal={arXiv preprint arXiv:2608.16157},
  year={2026}
}
```

## Acknowledgment

FreeToken was deeply inspired by [mini-sglang](https://github.com/sgl-project/mini-sglang), and
learned the design and reused code from the following projects:
[SGLang](https://github.com/sgl-project/sglang),
[vLLM](https://github.com/vllm-project/vllm),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer),
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention),
[LightLLM](https://github.com/ModelTC/lightllm) and [llama.cpp](https://github.com/ggml-org/llama.cpp).

## License

For the optional Qwen3.8 pinned-PLE long-prefill execution path, see
[layer-major prefill](docs/layer_major_prefill.md).

[Apache License 2.0](https://github.com/FlashML-org/FreeToken/blob/main/LICENSE).
