"""Qwen3.8-Flash-Next checkpoint reader (the NVFP4 and the official block-fp8 releases).

Three separate paths, because the checkpoint's three weight classes live in different places:

* :func:`iter_weights` -- every dense (non-expert) tensor, with the ``model.language_model.`` prefix stripped and fused where the model expects one buffer. See ``_DenseFuser``.
* :func:`load_ple_table` -- the 47.7 GiB FP8 n-gram table, 128 checkpoint shards concatenated into one pinned :class:`HostBank`.
* :func:`nvfp4_expert_spec` -- how the routed NVFP4 experts are named, for the offload cache's expert reader.

Dropped: ``mtp.*`` (speculative head, including its stacked ``mtp.layers.0.mlp.experts.*``); ``model.visual.*`` is kept only when the model built the tower.
"""

from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.config import VISION_KEY_PREFIXES
from freetoken.models.loader import drop_page_cache, iter_weight_files, shard_tensor
from freetoken.models.qwen3_vl.weight import rename_vl_prefix
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
)
from freetoken.layers.quantization import get_quant_config
from freetoken.models.register import get_model_spec
from freetoken.moe.host_banks import HostBank, read_range_into
from freetoken.utils import cached_load_hf_config, div_even, download_hf_weight
from freetoken.utils.progress import byte_bar
from tqdm import tqdm

# Routed NVFP4 experts (nvidia modelopt layout): per-expert, un-fused. Matched against the RAW
# weight_map key in nvfp4_banks. The ``model.language_model.`` anchor excludes the MTP head's
# stacked ``mtp.layers.N.mlp.experts.*`` tensors.
_EXPERT_KEY_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
)
_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: layer,  # every layer is MoE
    desc="Qwen3.8-Flash-Next NVFP4 experts",
)
# Per-tensor modelopt quant scales; consumed with their ``.weight`` (experts) or unused.
_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")

# The n-gram table itself: too big for the dense state dict, loaded by load_ple_table.
_PLE_TABLE_INFIX = ".ple.ple_embedding.ngram_embedding."
_PLE_SHARD_RE = re.compile(
    r"\.ple\.ple_embedding\.ngram_embedding\.shard_(?P<shard>\d+)\.weight$"
)
_PLE_SCALE_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight_scale"
_PLE_FILE_BYTES = 4 << 30  # ple-table-*.safetensors written by ftw_side_files

# Zero-centered Qwen4ExpTextRMSNorm weights, loaded RAW: GroupedPlusOneRMSNorm / GemmaPlusOneRMSNorm
# and the vendored grouped_gemma_rmsnorm all apply (1+w) at runtime in fp32, so folding the +1 into
# the bf16 weight here would double-apply it and round away small |w|. The GDN gated norm
# (linear_attn.norm) is a plain weight*x norm and is not in this set.
_ZERO_CENTERED_NORM_SUFFIXES = (
    ".hc_norm.weight",
    ".ple.norm_key.weight",
    ".ple.norm_query.weight",
    ".ple.norm_conv.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
    ".self_attn.indexer.q_layernorm.weight",
    ".self_attn.indexer.k_layernorm.weight",
)

# The per-layer HC mix reads the low-rank down projection and the injection logits from one GEMM; vLLM pads the merged rows to a multiple of 16 for cuBLAS (hyperconnection.py pad_size).
# The top-level hyper_connection_mixer has no injection and never fuses.
_PAD_TO = {"input_mix_weight_down_block_inject": 16}
_HC_WITH_INJECT = (".attn_hyper_connection", ".mlp_hyper_connection")
_KIND_SUFFIXES = (".weight_scale_inv", ".weight")
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_ELEM_DTYPES = {"e4m3": torch.float8_e4m3fn}


def _rename(raw_name: str) -> str | None:
    """Checkpoint key -> FreeToken state-dict key, or None to skip."""
    if raw_name.startswith("mtp."):
        return None
    if _PLE_TABLE_INFIX in raw_name:
        return None  # n-gram table + its scale: load_ple_table
    if _EXPERT_RE.search(raw_name):
        return None  # routed experts: offload source banks
    if raw_name.endswith(_SCALE_SUFFIXES):
        return None
    return rename_vl_prefix(raw_name)


def _split_kind(name: str) -> tuple[str, str]:
    """``name`` -> ``(module, kind)``; kind is "" for tensors that are neither a weight nor a block scale."""
    for suffix in _KIND_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix
    return name, ""


class _DenseFuser:
    """Concatenates checkpoint projection parts into the model's merged buffers, per kind (weight / block scale).

    The part table is the family's packed_modules_mapping. The QuantConfig picks the GDN in_proj layout and validates each part against the scheme the model built its buffer from.
    """

    def __init__(self, quant, packed: tuple[tuple[str, tuple[str, ...]], ...]) -> None:
        self.quant = quant
        self.groups = {fused: parts for fused, parts in packed if fused != "experts"}  # experts: bank reader
        self.by_part: dict[str, list[tuple[str, int]]] = {}
        for fused, parts in self.groups.items():
            for idx, part in enumerate(parts):
                self.by_part.setdefault(part, []).append((fused, idx))
        self.buf: dict[tuple[str, str], dict[int, torch.Tensor]] = {}

    def scheme(self, module: str):
        return None if self.quant is None else self.quant.scheme_for(module)

    def _target(self, parent: str, leaf: str) -> tuple[str, int] | None:
        candidates = self.by_part.get(leaf)
        if not candidates:
            return None
        if len(candidates) > 1:
            # GDN: quantized checkpoints split qkv|z from the bf16 b|a; same test as gdn.py
            split = self.scheme(f"{parent}.in_proj_qkvz") is not None
            keep = {"in_proj_qkvz", "in_proj_ba"} if split else {"in_proj"}
            candidates = [c for c in candidates if c[0] in keep]
            if not candidates:
                raise ValueError(f"{parent}.{leaf}: no merged projection for the {'split' if split else 'fused'} GDN layout")
        fused, idx = candidates[0]
        if fused in _PAD_TO and not parent.endswith(_HC_WITH_INJECT):
            return None
        return f"{parent}.{fused}", idx

    def check(self, module: str, name: str, tensor: torch.Tensor) -> None:
        """``tensor`` (checkpoint key ``name``) must match the scheme the model built ``module`` from."""
        scheme = self.scheme(module)
        if name.endswith(".weight_scale_inv"):
            if scheme is None or not scheme.has("weight_scale_inv"):
                raise ValueError(f"{name}: {module} has no block scale in the checkpoint's quant config ({scheme})")
            return
        is_fp8 = tensor.dtype in _FP8_DTYPES
        if scheme is None:
            if is_fp8:
                raise ValueError(f"{name} is {tensor.dtype} but the checkpoint's quant config declares {module} unquantized")
            return
        expected = _ELEM_DTYPES.get(scheme.weight.elem)
        if expected is not None and tensor.dtype is not expected:
            raise ValueError(f"{name} is {tensor.dtype} but the checkpoint's quant config declares {module} {scheme}")
        rows, cols = (scheme.weight.group or (1, 1))
        if rows > 1 and tensor.shape[0] % rows or cols > 1 and tensor.shape[1] % cols:
            raise ValueError(f"{name}: {tuple(tensor.shape)} is not a multiple of the {rows}x{cols} scale block of {module}")

    def check_unfused(self, name: str, tensor: torch.Tensor) -> None:
        module, kind = _split_kind(name)
        if kind == ".weight_scale_inv" or (kind == ".weight" and tensor.dtype in _FP8_DTYPES):
            self.check(module, name, tensor)

    def fuse(self, name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]] | None:
        """Buffer a part; return the merged ``[(name, tensor)]`` once its kind is complete, ``[]`` while incomplete, ``None`` if ``name`` is not a part."""
        module, kind = _split_kind(name)
        if not kind:
            return None
        parent, _, leaf = module.rpartition(".")
        hit = self._target(parent, leaf)
        if hit is None:
            return None
        fused, idx = hit
        self.check(fused, name, tensor)
        slots = self.buf.setdefault((fused, kind), {})
        slots[idx] = tensor
        parts = self.groups[fused.rpartition(".")[2]]
        if len(slots) < len(parts):
            return []
        del self.buf[(fused, kind)]
        rows = [slots[i] for i in range(len(parts))]
        pad_to = _PAD_TO.get(fused.rpartition(".")[2], 0) if kind == ".weight" else 0
        pad = (-sum(t.shape[0] for t in rows)) % pad_to if pad_to else 0
        if pad:
            rows.append(torch.zeros(pad, *rows[0].shape[1:], dtype=rows[0].dtype, device=rows[0].device))
        return [(fused + kind, torch.cat(rows, dim=0))]


def _shard_rows(
    t: torch.Tensor, parts: list[tuple[int, int]], rank: int, world: int
) -> torch.Tensor:
    """Column-parallel slice (dim 0) of a ``[part0 | part1 | ...]`` fusion; ``parts`` gives each
    part as ``(heads, rows_per_head)``. Heads split evenly across ranks; a part with fewer heads
    than ranks (GQA kv) replicates head ``rank * heads // world``, the ``div_even(...,
    allow_replicate=True)`` convention of the TP-aware layers."""
    out, off = [], 0
    for heads, rows in parts:
        local = div_even(heads, world, allow_replicate=True)
        first = rank * heads // world
        out.append(t[off + first * rows : off + (first + local) * rows])
        off += heads * rows
    assert off == t.shape[0], f"fusion parts {parts} cover {off} rows, tensor has {t.shape[0]}"
    return torch.cat(out, dim=0)


def _shard(name: str, t: torch.Tensor, config, rank: int, world: int) -> torch.Tensor:
    """TP shard of one state-dict tensor (fused projections included); identity at TP=1.

    Column-parallel (dim 0, by head): attention ``qkv_proj`` [q|gate per head | k | v], GDN
    ``in_proj`` [q | k | v | z | b | a] and the matching ``conv1d`` channels, ``A_log`` /
    ``dt_bias``, shared-expert ``gate_up_proj``. Row-parallel (dim 1): ``o_proj``,
    ``out_proj``, shared-expert ``down_proj``. Vocab rows: ``embed_tokens`` / ``lm_head``.
    Everything else (router, indexer, norms, HC, PLE, shared-expert gate) is replicated.

    The vision tower stays replicated (every rank builds its own tower); the block-fp8 dense
    layouts are not sharded and raise.
    """
    if world == 1:
        return t
    if name.startswith(VISION_KEY_PREFIXES):
        return t
    # _shard_rows slices rows, not 128x128 scale blocks, so a block-fp8 dense build cannot be sharded here.
    if name.endswith(".weight_scale_inv") or t.dtype in _FP8_DTYPES:
        raise NotImplementedError("qwen4_exp tensor parallelism serves bf16 dense weights, not block-fp8 ones")
    if name.endswith(".self_attn.qkv_proj.weight"):
        q = (config.num_qo_heads, 2 * config.head_dim)
        kv = (config.num_kv_heads, config.head_dim)
        return _shard_rows(t, [q, kv, kv], rank, world)
    if ".linear_attn." in name:
        g = config.linear_attention_group()
        k = (g.num_key_heads, g.key_head_dim)
        v = (g.num_value_heads, g.value_head_dim)
        if name.endswith(".in_proj.weight"):
            return _shard_rows(t, [k, k, v, v, (v[0], 1), (v[0], 1)], rank, world)
        if name.endswith(".conv1d.weight"):
            return _shard_rows(t, [k, k, v], rank, world)
        if name.endswith((".A_log", ".dt_bias")):
            return _shard_rows(t, [(v[0], 1)], rank, world)
        if name.endswith(".out_proj.weight"):
            return t.chunk(world, dim=1)[rank].clone()
        return t
    if name.endswith(".shared_expert.gate_up_proj.weight"):
        half = t.shape[0] // 2
        return _shard_rows(t, [(half, 1), (half, 1)], rank, world)
    # o_proj / down_proj: dim 1; embed_tokens / lm_head: vocab rows; others unchanged.
    return shard_tensor(name, t, rank=rank, world_size=world, num_kv_heads=None)


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    include_vision: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the dense (non-expert) weights, prefix-stripped and fused to the model's buffers.

    Keys keep the checkpoint's module names below the stripped prefix, so the emitted set is the model's state dict minus the routed experts.
    A dense projection is bf16 or 128x128 block-fp8 (``.weight`` e4m3 + ``.weight_scale_inv``) as the checkpoint's QuantConfig says: the official releases skip everything but the routed experts, the community NVFP4-FP8 requants quantize the attention / GDN projections.
    Fusions, per kind: attention q|k|v -> ``qkv_proj``; GDN ``in_proj_{qkv,z,b,a}`` -> ``in_proj``, or ``in_proj_qkvz`` + bf16 ``in_proj_ba`` when qkv|z is quantized; shared-expert gate|up -> ``gate_up_proj``; each per-layer HC's ``input_mix_weight_down`` | ``block_inject_weight`` -> a zero-padded ``input_mix_weight_down_block_inject``.
    ``include_moe_experts`` is accepted for the loader contract but never yields anything: the routed experts are NVFP4 and always come from the offload cache's expert reader.
    """
    if not include_non_moe:
        return

    from .config import parse_config

    hf_config = cached_load_hf_config(model_path)
    spec = get_model_spec(hf_config.architectures[0])
    fuser = _DenseFuser(get_quant_config(), spec.packed_modules_mapping)
    tp = get_tp_info()
    # _shard slices by head counts, which only the parsed config carries.
    config = parse_config(hf_config) if tp.size > 1 else None
    for file in tqdm(
        iter_weight_files(model_path),
        desc="Loading weights",
        disable=not tp.is_primary(),
    ):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _rename(raw_name)
                if name is None:
                    continue
                if not include_vision and name.startswith(VISION_KEY_PREFIXES):
                    continue
                tensor = f.get_tensor(raw_name)
                fused = fuser.fuse(name, tensor)
                if fused is None:
                    fuser.check_unfused(name, tensor)
                    yield name, _shard(name, tensor, config, tp.rank, tp.size)
                else:
                    for fused_name, fused_tensor in fused:
                        yield fused_name, _shard(fused_name, fused_tensor, config, tp.rank, tp.size)

    assert not fuser.buf, f"Incomplete projection fusions: {sorted(k[0] + k[1] for k in fuser.buf)}"


def iter_vision_weights(model_path: str, device: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    """The vision tower alone, named as iter_weights names it."""
    for file in iter_weight_files(model_path):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _rename(raw_name)
                if name is not None and name.startswith(VISION_KEY_PREFIXES):
                    yield name, f.get_tensor(raw_name)


# ======================================================================================
# PLE n-gram table
# ======================================================================================


@dataclass(frozen=True)
class PleTable:
    """The filled n-gram table: one pinned host bank plus the checkpoint's per-tensor FP8 scale."""

    bank: HostBank
    weight_scale: torch.Tensor  # scalar, checkpoint dtype (bf16)

    @property
    def tensor(self) -> torch.Tensor:
        """``[total_rows, ngram_head_dim]`` float8_e4m3fn view of the bank."""
        return self.bank.tensor


_PLE_ST_DTYPE = "F8_E4M3"


def _safetensors_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def _ple_table_files(folder: str) -> list[str]:
    """Shards holding a piece of the n-gram table, from the index when there is one."""
    index = os.path.join(folder, "model.safetensors.index.json")
    if not os.path.exists(index):
        return sorted(iter_weight_files(folder))
    with open(index, encoding="utf-8") as fh:
        weight_map = json.load(fh)["weight_map"]
    files = {shard for name, shard in weight_map.items() if _PLE_TABLE_INFIX in name}
    return sorted(os.path.join(folder, shard) for shard in files)


def ftw_side_files(model_path: str, out_dir: str) -> list[str]:
    """Write the PLE n-gram table tensors, and only those, into ``ple-table-*.safetensors`` next to an FTW checkpoint.

    The table is served from safetensors files in the checkpoint dir (see load_ple_table), not from FTW entries."""
    from safetensors.torch import save_file

    folder = download_hf_weight(model_path)
    written: list[str] = []
    batch: dict[str, torch.Tensor] = {}
    size = 0

    def flush():
        nonlocal batch, size
        if batch:
            name = f"ple-table-{len(written):05d}.safetensors"
            save_file(batch, os.path.join(out_dir, name))
            written.append(name)
            batch, size = {}, 0

    for path in _ple_table_files(folder):
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if _PLE_TABLE_INFIX not in key:
                    continue
                t = f.get_tensor(key)
                batch[key] = t
                size += t.numel() * t.element_size()
                if size >= _PLE_FILE_BYTES:
                    flush()
    flush()
    return written


def load_ple_table(model_path: str, qwen4_args, *, pin: bool = True,
                   workers: int = 8, chunk: int = 8 << 20) -> PleTable:
    """Concatenate the checkpoint's ``ngram_embedding.shard_<i>`` tensors into one pinned host bank.

    The checkpoint splits the table into ``split_ngram_parts`` equal row blocks named by shard
    index and scattered over the ``model-plefp8-*`` shards in header (lexicographic) order, so the
    bank is filled shard by shard at ``shard_index * rows_per_shard``. Each read is O_DIRECT: the
    table is ~47.7 GiB and must not also sit in the page cache while the bank holds the same bytes.
    """
    folder = download_hf_weight(model_path)
    parts: dict[int, tuple[str, int, int]] = {}  # shard index -> (path, file offset, bytes)
    scale: torch.Tensor | None = None
    rows = cols = 0
    for path in _ple_table_files(folder):
        header, base = _safetensors_header(path)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if key.endswith(_PLE_SCALE_SUFFIX):
                with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                    scale = f.get_tensor(key).reshape(())
                continue
            match = _PLE_SHARD_RE.search(key)
            if match is None:
                continue
            if meta["dtype"] != _PLE_ST_DTYPE:
                raise ValueError(f"PLE table shard {key} has unsupported dtype {meta['dtype']}")
            shape = meta["shape"]
            if rows and tuple(shape) != (rows, cols):
                raise ValueError(f"PLE table shard {key} is {shape}, expected {[rows, cols]}")
            rows, cols = shape
            begin, end = meta["data_offsets"]
            parts[int(match.group("shard"))] = (path, base + begin, end - begin)

    expected = int(qwen4_args.split_ngram_parts)
    if sorted(parts) != list(range(expected)):
        raise ValueError(
            f"PLE table needs shards 0..{expected - 1}, found {len(parts)}: {sorted(parts)[:8]}"
        )
    if cols != qwen4_args.ngram_head_dim:
        raise ValueError(f"PLE table row is {cols} wide, config says {qwen4_args.ngram_head_dim}")
    if scale is None:
        raise ValueError("PLE table has no weight_scale")

    bank = HostBank((expected * rows, cols), torch.float8_e4m3fn)
    shard_bytes = rows * cols
    bar = byte_bar(expected * shard_bytes, "Loading PLE table")
    try:
        buf = bank.memoryview()
        for shard in range(expected):
            path, offset, nbytes = parts[shard]
            assert nbytes == shard_bytes, f"PLE shard {shard} is {nbytes} B, expected {shard_bytes}"
            read_range_into(buf, path, file_offset=offset, nbytes=nbytes,
                            dest_offset=shard * shard_bytes, workers=workers, chunk=chunk)
            bar.update(nbytes)
    finally:
        bar.close()
    if pin and torch.cuda.is_available():
        bank.pin()
    return PleTable(bank=bank, weight_scale=scale)


# ======================================================================================
# Routed NVFP4 experts
# ======================================================================================


def nvfp4_expert_spec(model_path: str, config):
    return _NVFP4_SOURCE_SPEC


__all__ = [
    "nvfp4_expert_spec",
    "PleTable",
    "iter_weights",
    "load_ple_table",
]
