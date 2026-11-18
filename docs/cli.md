# CLI reference

```
ft <command> [args]
```

| Command | Purpose |
|---|---|
| `ft serve` | Start the API server (OpenAI `/v1/*`, Anthropic `/v1/messages`, Responses) |
| `ft shell` | Chat with a server in the terminal |
| `ft ctl` | Query and manage a running server over HTTP |
| `ft launch` | Configure and launch a coding agent against a server |
| `ft checkpoint` | Convert an HF checkpoint to the FTW fast-load format |
| `ft bench bw` | Benchmark CPU vs PCIe bandwidth to calibrate hybrid MoE decode |

`ft --version` prints the installed version (torch-free; nightly wheels carry a
`+g<sha>` build stamp, tagged releases a bare version). Every command supports
`--help`.

## ft serve

```bash
ft serve --model <path-or-hf-id> [options]
```

`--model` is the only required flag — dtype, attention backend, MoE backend,
MoE cache size, KV capacity, CUDA-graph sizes and the tool-call/reasoning
parsers all resolve automatically from the checkpoint and the GPU.

### Model

| Flag | Default | Meaning |
|---|---|---|
| `--model-path`, `--model` | required | Local dir, HF repo id, a `.gguf` file, or an FTW dir (auto-detected) |
| `--served-model-name` | basename of `--model` | Model id reported by `/v1/models` |
| `--hf-overrides` | — | JSON object applied to the checkpoint's config as vLLM's `--hf-overrides`: a nested config section updates key by key, any other value is replaced whole. A YaRN `rope_parameters` override serves `original_max_position_embeddings * factor` positions |

### Server & runtime

| Flag | Default | Meaning |
|---|---|---|
| `--host` | 127.0.0.1 | Bind address |
| `--port` | 1919 | Bind port |
| `--api-key` | `$FREETOKEN_API_KEY`, else off | Require the key on every route except `/health`, as `Authorization: Bearer <key>` or `x-api-key: <key>`; other requests get a 401 |
| `--gpu` | GPU 0 | GPU to run on: a UUID from `nvidia-smi -L` or an `nvidia-smi` index; see [below](#choosing-a-gpu) |
| `--max-running-requests` | 4 | Max concurrently running requests |
| `--max-output-tokens` | 32768 | Default output budget for requests that omit one |
| `--max-seq-len-override` | from checkpoint | Max sequence length |
| `--max-prefill-length` | 8192 | Chunked-prefill chunk size in tokens; with offloaded experts see [performance/prefill.md](performance/prefill.md) |
| `--cuda-graph-max-bs`, `--graph` | = max running requests | Max batch size captured as CUDA graphs |
| `--decode-log-interval` | 40 | Scheduler status line every N decode steps |

### Choosing a GPU

For example, a machine with an RTX 5090 and an RTX 3060 Ti:

```console
$ nvidia-smi -L
GPU 0: NVIDIA GeForce RTX 3060 Ti (UUID: GPU-2f3a9b1c-8d7e-4a05-b6c1-0e5f9a3d7b42)
GPU 1: NVIDIA GeForce RTX 5090 (UUID: GPU-9e8d7c6b-5a49-4f13-8207-c1b0a4e6d3f5)
```

```bash
ft serve --model ... --gpu 1             # by nvidia-smi index -- the 5090
ft serve --model ... --gpu GPU-9e8d7c6b  # the same card by UUID (a unique prefix is enough)
```

### KV cache & memory

| Flag | Default | Meaning |
|---|---|---|
| `--memory-ratio` | 0.9 | Fraction of free VRAM the engine may use (weights + MoE cache + KV) |
| `--num-pages` / `--num-tokens` | auto | KV capacity override in pages / tokens (mutually exclusive; auto sizes from VRAM left after weights and MoE cache) |
| `--embed-weights` | gpu | Where the input token embedding tables live. `host` keeps them in pinned host RAM and each lookup reads its rows over PCIe, so their VRAM goes to the expert cache and KV. A table tied to the LM head stays on the GPU, and `host` is refused when there is no other table; Gemma-4 E2B / E4B's per-layer table is the one that moves, the way those models fit an 8 GB GPU |
| `--online-quant` | — | Convert the Linear layers and input embedding tables that the checkpoint stores in bf16 at load. `df11` compresses them losslessly to about 11 bits per weight and decodes them bit-for-bit on use: about 30% less VRAM for those weights, at the cost of a decode per use. Layers the checkpoint quantizes, the LM head, a table tied to it, vision encoders and layers whose weight the model reads directly are left as they are, and so are tables under TP > 1; refused when no layer qualifies |
| `--online-quant-modules` | all | Limit `--online-quant` to these modules: comma-separated checkpoint module names or glob patterns, e.g. `model.layers.*.self_attn,model.embed_tokens` (`model.language_model.layers.*` where the checkpoint nests the text model); a name also covers the modules under it, and one that selects no convertible layer is refused. A fused layer (`qkv_proj`) must be covered in all of its checkpoint parts |
| `--page-size` | 1 | KV page size; DSV4 forces 128, the TRTLLM backend needs 16/32/64, SWA models require 1 |
| `--embed-device` | gpu | `cpu` keeps the input-embedding table in pinned host RAM; the GPU reads the looked-up rows over PCIe, and the freed VRAM goes to the KV cache (2.37 GiB on Qwen3.8-27B). Decode reads one row per token, so speed is unchanged. A table tied to the LM head stays on the GPU |
| `--cache-type` | radix | `radix` (prefix reuse; SWA/GDN-aware variants picked automatically) or `naive` |
| `--kv-cache-dtype` | bf16 | `bf16` or `fp8`: store the KV cache as e4m3 codes plus one fp32 scale per (token, kv head), roughly doubling the tokens that fit in the same VRAM; see [FP8 KV cache](#fp8-kv-cache) |
| `--attention-backend`, `--attn` | auto | `trtllm`/`fi`/`fa`/`triton`/`dsv4_sparse`/`dsa`; `prefill,decode` pair allowed; auto picks per model + GPU |
| `--kv-cache-dtype` | auto | How K and V are stored. `auto` keeps the compute dtype. `fp8_per_token_head` (e4m3) and `int8_per_token_head` store 8 bits per element plus one fp32 scale per token and head (two per MLA latent row, for ckv and kpe), about half the KV memory. The `triton`, `qsa_sparse`, `m3_sparse` and `dsa` backends read them; `auto` picks one of these. They need `--dtype bfloat16` and no tensor parallelism. An `mtp` draft stores its KV in the same format |

### FP8 KV cache

`ft serve --kv-cache-dtype fp8` halves the bytes per cached token (8-bit codes instead
of 16), so a card that held N tokens holds close to 2N. Each `(token, kv head)` row
keeps its own fp32 scale, which costs ~3% back at `head_dim=128`. Requirements and
trade-offs:

- Needs the **triton** attention backend; `--attn auto` selects it (and refuses an
  explicit `fi`/`fa`/`trtllm`, which cannot be shown to apply these scales).
- Works on the plain paged, hybrid-SWA and QSA sparse (Qwen3.8-Flash-Next) KV pools.
  On QSA the block-selection index keys stay 16-bit; only the selected K/V rows are
  read back as codes. MLA/DSA latent KV, DeepSeek-V4's tiered pool and the block-sparse
  MiniMax-M3 pool stay 16-bit; asking for fp8 there fails at startup rather than
  silently ignoring the flag.
- The same bytes on every GPU FreeToken targets: the codes sit in a plain byte buffer
  and are decoded in software, so the cache holds identical data and produces identical
  numbers on any card (the fp8 type is deliberately kept out of the kernels, which is
  also what makes the feature work on the RTX 30 series).
- Qwen3.8 FP8-QSA uses the fast scaled16 read path by default. It keeps decoded FP8
  tiles in fp16 and applies each row scale around the two dot products, substantially
  reducing prefill cost on FP8 KV. Set `FREETOKEN_QSA_FP8_FAST_SCALE=0` to select the
  legacy, bit-exact reconstruction; the fast path changes final BF16 rounding order.
- Accuracy is checkpoint-dependent. Expect it to matter most on long contexts and on
  models with outlier key channels; keep `bf16` when a run must be bit-reproducible.
- `ft ctl stats` / `/v1/cache/status` report the smaller `kv_bytes_per_token`, and
  `ft ctl cache --kv N` moves the same (now cheaper) pool.

### MoE offload

See [models.md](models.md#moe-strategies) for what each strategy does, [performance/prefill.md](performance/prefill.md) for prefill speed, and [performance/decode.md](performance/decode.md) for decode speed.

| Flag | Default | Meaning |
|---|---|---|
| `--moe-strategy` | auto | `fused`/`offload`/`cpu`/`hybrid`; auto → offload; fused on unified-memory GPUs (GB10), see [models.md](models.md#unified-memory-gpus-gb10--dgx-spark). `--moe-backend` is the deprecated old spelling |
| `--quant-backend` | auto | Kernel per quantized layer type, `layer[.kind]=name` entries: `linear=marlin,moe=b12x` or `moe.nvfp4=triton`. A layer-level entry applies to every kind whose table lists the name |
| `--nvfp4-backend` | — | Deprecated: stands in for `--quant-backend moe.nvfp4=<marlin\|b12x\|triton>` (`flashinfer` means b12x); cannot be combined with `--quant-backend` |
| `--moe-cache-size` / `--moe-cache-rate` / `--moe-cache-mem` / `--moe-cache-auto` | auto | GPU expert-cache size as slots / fraction of all experts / memory such as `12GiB` / sized from free VRAM (mutually exclusive; auto is enabled by default for offload-family strategies) |
| `--moe-cache-scope` | unified | `unified`: one LRU pool of slots per expert quant (quant type and bank layout), so all MoE layers share one pool when they share a quant format; `per-layer`: each layer has its own pool. `per-layer` does not combine with `--moe-strategy cpu` or `--moe-cpu-layers` |
| `--moe-cache-split` | equal | How the pools share the slots: `equal` (in proportion to their layer counts), or `profile:<path>`: every pool keeps `top_k` slots and the rest go to the most used experts in the profile's per-layer expert counts. With `offload`, a pool smaller than the expert count lowers `--max-running-requests` to its slots / top_k |
| `--kv-reserve-tokens` | 8192 | KV token floor reserved before `--moe-cache-auto` fills experts |
| `--moe-cpu-threads` | physical cores | CPU worker threads for the cpu/hybrid executor |
| `--moe-cpu-layers` | all on GPU | With `offload`/`hybrid`: which MoE layers decode on CPU (`3,7,11`, a count, a fraction, or `auto`). `auto` is for hosts that cap CUDA pinned memory (Windows/WSL): it pins layers from the middle out until CUDA refuses one and OS-locks the rest (`FREETOKEN_PIN_BUDGET_GB` and FTW checkpoints lock from a size estimate instead), see [models.md](models.md#hosts-that-cap-pinned-memory-wsl); every value needs an expert format the CPU executor serves (bf16, fp8_block, nvfp4, mxfp4, GGUF on x86 CPUs, except gpt-oss GGUF files whose per-expert biases decode on the GPU only) |
| `--moe-hybrid-max-fetch` | auto | With `hybrid`: max experts fetched over PCIe per layer per step; rest computed on CPU |
| `--moe-prefill-demand-tokens` | 0 | Prefill: a chunk of at most this many tokens loads only the experts it routes to, through the expert cache as decode does (GPU decode only); 0 streams whole layers |
| `--moe-locked-fetch` | auto | With `hybrid` and locked layers: share of a locked layer's misses fetched per step into its GPU slots through pinned staging. Auto takes p / (2p + c) from the `ft bench bw` row of the layer's kind (at most 1 per step without a row); 0 = locked layers decode on CPU only |
| `--moe-locked-max-fetch` | 8 | Most experts a locked layer fetches per step; sizes the staging (that many rows of the widest expert, per bank) |
| `--moe-collect-stats` | off | Log decode MoE stats with each decode log line: hit rate, and per layer-step the misses, experts fetched and experts computed on CPU, for pinned, locked and CPU-only layers |
| `--moe-prefill-hit-d2d` | off | Prefill: copy cache-hit experts device-side, stream only misses (CUDA >= 13) |
| `--disable-moe-prefill-overlap` | overlap on | Disable the two-buffer prefill copy overlap |

### Speculative decoding

| Flag | Default | Meaning |
|---|---|---|
| `--speculative-algorithm`, `--spec-algo` | off | Draft method. `mtp` drafts with the checkpoint's multi-token-prediction layer: Qwen3.5 / Qwen3.6 HF or GGUF checkpoints, and Qwen3.8-Flash-Next HF or GGUF; Gemma-4 (dense, E2B / E4B and MoE) keeps no draft in the target file and drafts from an `mtp-*.gguf` assistant sidecar (architecture `gemma4-assistant`) given with `--speculative-draft-model-path` / `--spec-draft-model`. `dflash` drafts a block of tokens per step with a DFlash 2 draft checkpoint over the target's layer outputs, for the Qwen3.5 / Qwen3.6 / Qwen3.8 family and Qwen3.8-Flash-Next: `incoai/Qwen3.8-27B-DFlash2` for Qwen3.8-27B, `incoai/Qwen3.6-35B-A3B-DFlash2` for Qwen3.6-35B-A3B, `incoai/Qwen3.8-Flash-Next-DFlash2` for Qwen3.8-Flash-Next (or its GGUF release, `incoai/Qwen3.8-Flash-Next-DFlash2-GGUF`, as BF16, Q4_K_M, Q5_K_M, Q6_K or Q8_0) |
| `--speculative-num-steps`, `--spec-num-steps` | the method's default (3 for `mtp`, the draft's `block_size` - 1 for `dflash`) | Draft tokens per decode step, 1 to 8 for `mtp`; 1 to the draft's `block_size` - 1 for `dflash` |
| `--speculative-draft-model-path`, `--spec-draft-model` | the target checkpoint | Where the draft weights load from. A Qwen3.8-Flash-Next target also takes a Strata MTP pack (a GGUF that keeps the checkpoint's `mtp.*` tensors under their HF names), and a Qwen3.8-Flash-Next GGUF target needs one; a Qwen3.5 / Qwen3.6 GGUF target drafts from its own `nextn` MTP block (the block llama.cpp keeps after the last decoder layer), or from an `mtp-*.gguf` sidecar of the same model (unsloth's `MTP/` releases) given here; for `mtp`, a separate checkpoint folder is an HF-target option, read with the target's quantization config, and a GGUF target refuses a folder draft. `dflash` requires it: the DFlash 2 draft checkpoint (bf16 safetensors with its `config.json`, or a local `.gguf` from `incoai/Qwen3.8-Flash-Next-DFlash2-GGUF`, served natively at its BF16, Q4_K_M, Q5_K_M, Q6_K or Q8_0 types), whose `target_layer_ids`, hidden size and vocabulary must match the target (`num_target_layers` too, when the checkpoint records it; only the HF checkpoint does) |
| `--speculative-token-map`, `--spec-token-map` | off | `mtp` only. A file of distinct little-endian int32 token ids (`python scripts/make_token_map.py MODEL` writes one): the draft computes its logits over those LM head rows alone, from a copy made at startup whose size is logged. That VRAM comes out of the expert cache when `--moe-cache-auto` sizes it (the default for hybrid and offload without a cache size), and out of the KV cache otherwise. The target still verifies over the whole vocabulary, so the output distribution does not depend on the file, only the acceptance rate does. A tied or block-FP8 LM head is refused |

Speculative decoding supports tensor parallelism (`--tp`) with replicated drafts and cross-rank broadcast consensus, and runs without overlap scheduling; a target with sliding-window KV is served only by the Gemma-4 assistant, which reads the target's cache through a read-only view; every other target with sliding-window KV is refused at startup, and so is every target with DSV4 window-tier KV. The Gemma-4 assistant keeps no KV of its own: it reads the target's cache, so it adds nothing to the KV budget and its weights cost about 149 MiB for the E4B draft, about 425-428 MiB for the 12b and 26B-A4B Q8_0 ones. `mtp` refuses an FTW checkpoint as the draft source, since FTW drops the `mtp.*` tensors, and a Qwen3.5 / Qwen3.6 GGUF converted without its MTP layer (llama.cpp `--no-mtp`) when no `mtp-*.gguf` sidecar is given: such a trunk has no draft of its own to read, while a `--no-mtp` trunk plus a sidecar works (the trunk config's shapes, the sidecar's weights).

`dflash` keeps the draft's context in a per-request ring of the draft's sliding window (2048 positions for the released drafts) and, on a hybrid GDN target, in a window on the GDN state slots, so a prefix-cache hit restores it with the GDN snapshot; the window adds `(window + 64) x hidden` activation-dtype bytes to every GDN slot (`/v1/cache/status` counts them). A target without GDN layers needs `--cache-type naive`. DFlash v1 drafts are refused, and so is a quantized safetensors draft: the GGUF release is the quantized path. A quantized GGUF draft loads its linear weights at the file's own ggml types (Q4_K_M, Q5_K_M, Q6_K or Q8_0) and dequantizes only the selector codebooks and the K/V context copy; a BF16/F16/F32 GGUF loads floats throughout. With sampling, the drafts are drawn from the draft's own distribution over its top-16 candidates and verified by rejection sampling against the target's distribution, so the output distribution is the target's.

`dflash`