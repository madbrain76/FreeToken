"""Qwen3.8-Flash-Next checkpoint reader (bf16, block-fp8, ModelOpt and llm-compressor exports).

Three separate paths, because the checkpoint's three weight classes live in different places:

* :func:`iter_weights` -- every dense (non-expert) tensor, with the ``model.language_model.`` prefix stripped and fused where the model expects one buffer. See ``_DenseReader``.
* :func:`load_ple_table` -- the n-gram table (47.7 GiB fp8, 25.5 GiB packed nvfp4, or 95.4 GiB bf16), 128 checkpoint shards concatenated into one pinned :class:`HostBank`.
* :func:`nvfp4_expert_spec` -- how the routed NVFP4 experts are named under the checkpoint's dialect, for the offload cache's expert reader.

Dropped: ``mtp.*`` (speculative head, read by :func:`iter_mtp_weights`); ``model.visual.*`` is kept only when the model built the tower.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import re
import struct
from typing import Iterator

import safetensors
import torch

from freetoken.distributed import get_tp_info
from freetoken.models.loader import drop_page_cache, iter_weight_files
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
)
from freetoken.layers.quantization import QuantConfig, QuantKind, QuantScheme, get_quant_config
from freetoken.models.config import VISION_KEY_PREFIXES
from freetoken.models.qwen3_vl.weight import rename_vl_prefix
from freetoken.models.register import get_model_spec
from freetoken.moe.host_banks import HostBank, read_range_into
from freetoken.utils import cached_load_hf_config, div_even, download_hf_weight
from freetoken.utils.progress import byte_bar
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Routed NVFP4 experts: per-expert, un-fused. Matched against the RAW weight_map key in nvfp4_banks.
# The ``model.language_model.`` anchor excludes the MTP head's stacked ``mtp.layers.N.mlp.experts.*`` tensors.
_EXPERT_KEY_RE = (
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>{kinds})$"
)
_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
# role -> the expert bank reader's canonical (ModelOpt) tensor kind
_BANK_KINDS = {"weight": "weight", "weight_scale": "weight_scale", "weight_global": "weight_scale_2"}

# leaves the model builds as quant-aware Linear layers: their tensors are read under the QuantConfig
_LINEAR_LEAVES = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj", "index_qk_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    "gate_proj", "up_proj", "down_proj",
    "input_mix_weight_down", "block_inject_weight", "input_mix_weight_up",
    "key_proj", "value_proj", "lm_head", "fc_embedding", "fc_hidden",
})
# activation scales of modules whose scheme carries no input_scale role
_DROPPED_SUFFIXES = frozenset({"input_scale", "input_global_scale"})
_ELEM_DTYPES = {"e4m3": torch.float8_e4m3fn, "e2m1": torch.uint8}
_QUANT_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8, torch.int8)

# The n-gram table itself: too big for the dense state dict, loaded by load_ple_table.
_PLE_TABLE_INFIX = ".ple.ple_embedding.ngram_embedding."
_PLE_SCALE_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight_scale"
_PLE_SHARD_RE = re.compile(
    r"\.ple\.ple_embedding\.ngram_embedding\.shard_(?P<shard>\d+)\.weight$"
)

# Gemma-style zero-centered RMSNorm: the checkpoint stores the offset `w` rather than the
# runtime multiplier `1 + w`. The engine models have that +1 folded into their RMSNorm.
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

# Layers whose fused projections pad rows to a multiple of 16 so Triton kernels stay aligned.
# The top-level hyper_connection_mixer has no injection and never fuses.
_PAD_TO = {"input_mix_weight_down_block_inject": 16}
_HC_WITH_INJECT = (".attn_hyper_connection", ".mlp_hyper_connection")


def _rename(raw_name: str) -> str | None:
    """Checkpoint key -> FreeToken state-dict key, or None to skip."""
    if raw_name.startswith("mtp."):
        return None
    if _PLE_TABLE_INFIX in raw_name:
        return None  # n-gram table + its scale: load_ple_table
    if _EXPERT_RE.search(raw_name):
        return None  # routed experts: offload source banks
    # static KV-cache scales of the quantizers; the KV cache runs in the engine's dtype
    if raw_name.endswith((".k_scale", ".v_scale", ".q_scale", ".prob_scale")):
        return None
    return rename_vl_prefix(raw_name)


def _per_row_scale(scale: torch.Tensor, rows: int) -> torch.Tensor:
    """Per-tensor scalar or per-channel ``[rows, 1]`` scale -> fp32 ``[rows]``; any other count is refused rather than broadcast onto the wrong rows."""
    flat = scale.reshape(-1).to(torch.float32)
    if flat.numel() == 1:
        return flat.expand(rows).contiguous()
    if flat.numel() != rows:
        raise ValueError(f"scale has {flat.numel()} elements for a weight with {rows} output rows (shape {tuple(scale.shape)}); expected 1 or {rows}")
    return flat.contiguous()


class _DenseReader:
    """Routes each Linear tensor to the buffer its module's scheme declares; packed projections are concatenated per role once every part is in.

    The part table is the family's packed_modules_mapping. The QuantConfig picks the GDN in_proj layout and, through its dialect's storage names, which checkpoint tensor fills which role.
    """

    def __init__(self, quant: QuantConfig | None, packed: tuple[tuple[str, tuple[str, ...]], ...]) -> None:
        self.quant = quant
        self.groups = {fused: parts for fused, parts in packed if fused != "experts"}  # experts: bank reader
        self.by_part: dict[str, list[tuple[str, int]]] = {}
        for fused, parts in self.groups.items():
            for idx, part in enumerate(parts):
                self.by_part.setdefault(part, []).append((fused, idx))
        # target module -> (part count, {part: {role: tensor}}, {part: the roles its module stores}, scheme)
        self.pending: dict[str, tuple[int, dict[int, dict[str, torch.Tensor]], dict[int, set[str]], QuantScheme | None]] = {}

    def scheme(self, module: str) -> QuantScheme | None:
        return None if self.quant is None else self.quant.scheme_for(module)

    def target(self, module: str) -> tuple[str, int, int]:
        """``(fused module, part index, part count)``; a standalone linear is its own single-part target."""
        parent, _, leaf = module.rpartition(".")
        candidates = self.by_part.get(leaf)
        if not candidates:
            return module, 0, 1
        if len(candidates) > 1:
            # GDN: quantized checkpoints split qkv|z from b|a; same test as layers/linear_attn/gdn.py
            split = self.scheme(f"{parent}.in_proj_qkvz") is not None
            keep = {"in_proj_qkvz", "in_proj_ba"} if split else {"in_proj"}
            candidates = [c for c in candidates if c[0] in keep]
        fused, idx = candidates[0]
        if fused in _PAD_TO and not parent.endswith(_HC_WITH_INJECT):
            return module, 0, 1
        return f"{parent}.{fused}", idx, len(self.groups[fused])

    def add(self, name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]] | None:
        """Take one tensor; the emitted ``[(name, tensor)]`` once its target module is complete, ``[]`` before, None if ``name`` is not a Linear's tensor."""
        module, _, suffix = name.rpartition(".")
        if module.rpartition(".")[2] not in _LINEAR_LEAVES:
            return None
        scheme = self.scheme(module)
        roles = {"weight": "weight"} if scheme is None else {n: r for r, e in self.quant.storage(scheme).items() for n in e.names}
        role = roles.get(suffix)
        if role is None:
            if suffix in _DROPPED_SUFFIXES:
                return []
            raise ValueError(f"{name}: the checkpoint's quant config declares {module} {scheme or 'unquantized'}, stored as {sorted(roles)}")
        if role == "weight":
            self._check_weight(name, module, scheme, tensor)
        target, idx, count = self.target(module)
        _, parts, expected, _ = self.pending.setdefault(target, (count, {}, {}, scheme))
        part = parts.setdefault(idx, {})
        if role in part:
            raise ValueError(f"{name}: {module} already has its {role} from another checkpoint tensor")
        part[role] = tensor
        expected[idx] = set(roles.values())
        if scheme is not None and set(part) == expected[idx]:
            parts[idx] = self._finish(module, scheme, part)
        if len(parts) < count or any(set(parts[i]) != expected[i] for i in parts):
            return []
        del self.pending[target]
        return self._emit(target, [parts[i] for i in range(count)], scheme)

    def missing(self) -> list[str]:
        """One line per incomplete module: the roles its parts still lack."""
        lines = []
        for target, (count, parts, expected, _scheme) in sorted(self.pending.items()):
            lacking = sorted(set().union(*(expected[i] - set(parts[i]) for i in parts)))
            if len(parts) < count:
                lacking.append(f"{count - len(parts)} of {count} fused parts")
            lines.append(f"{target}: missing {lacking}")
        return lines

    @staticmethod
    def _check_weight(name: str, module: str, scheme: QuantScheme | None, tensor: torch.Tensor) -> None:
        if scheme is None:
            if tensor.dtype in _QUANT_DTYPES:
                raise ValueError(f"{name} is {tensor.dtype} but the checkpoint's quant config declares {module} unquantized")
            return
        elem = _ELEM_DTYPES.get(scheme.weight.elem)
        if elem is None:
            raise ValueError(f"{name}: qwen4_exp has no dense reader for {scheme}")
        if tensor.dtype is not elem:
            raise ValueError(f"{name} is {tensor.dtype} but the checkpoint's quant config declares {module} {scheme}")
        rows, cols = tensor.shape[0], tensor.shape[1] * (2 if scheme.weight.elem == "e2m1" else 1)
        block_rows, block_cols = scheme.weight.group or (1, 1)
        if block_rows > 1 and rows % block_rows or block_cols > 1 and cols % block_cols:
            raise ValueError(f"{name}: {rows}x{cols} is not a multiple of the {block_rows}x{block_cols} scale block of {module}")

    def _finish(self, module: str, scheme: QuantScheme, part: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Validate one complete part against ``scheme`` and put its scales in the layer's form."""
        storage = self.quant.storage(scheme)
        part = {role: 1.0 / t.to(torch.float32) if storage[role].reciprocal else t for role, t in part.items()}
        weight = part["weight"]
        rows, cols = weight.shape[0], weight.shape[1] * (2 if scheme.weight.elem == "e2m1" else 1)
        block_rows, block_cols = scheme.weight.group or (1, 1)
        scale_role = "weight_scale_inv" if "weight_scale_inv" in part else "weight_scale"
        out = dict(part)
        if block_cols < 0:
            try:
                out[scale_role] = _per_row_scale(part[scale_role], rows)
            except ValueError as exc:
                raise ValueError(f"{module}.{scale_role}: {exc}") from None
        else:
            want = (rows // block_rows, cols // block_cols)
            if tuple(part[scale_role].shape) != want:
                raise ValueError(f"{module}: {scale_role} is {tuple(part[scale_role].shape)}, expected {want} for {scheme}")
            if scheme.weight.scale == "e4m3" and part[scale_role].dtype is not torch.float8_e4m3fn:
                raise ValueError(f"{module}: {scale_role} is {part[scale_role].dtype} but {scheme} stores e4m3 scales")
        if "weight_global" in part:
            g = part["weight_global"].reshape(-1).to(torch.float32)
            if g.numel() != 1:
                raise ValueError(f"{module}: weight_global has {g.numel()} elements, expected one per-tensor scale")
            out["weight_global"] = g.expand(rows).contiguous()
        if "input_scale" in part:
            out["input_scale"] = part["input_scale"].reshape(()).to(torch.float32)
        return out

    @staticmethod
    def _emit(target: str, parts: list[dict[str, torch.Tensor]], scheme: QuantScheme | None) -> list[tuple[str, torch.Tensor]]:
        pad_to = _PAD_TO.get(target.rpartition(".")[2], 0)
        if pad_to and scheme is not None:
            raise ValueError(f"{target}: qwen4_exp pads this fusion to {pad_to} rows and reads it bf16 only, the checkpoint's quant config declares {scheme}")
        out = []
        for role in parts[0]:
            tensors = [part[role] for part in parts]
            if role == "input_scale":
                # fused parts read the same activation, so one range covers them: max is exact then and safe if they drift
                out.append((f"{target}.{role}", torch.stack(tensors).max()))
                continue
            pad = (-sum(t.shape[0] for t in tensors)) % pad_to if pad_to else 0
            if pad:
                tensors.append(torch.zeros(pad, *tensors[0].shape[1:], dtype=tensors[0].dtype, device=tensors[0].device))
            out.append((f"{target}.{role}", tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)))
        return out


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
    """Slice projection weights for tensor parallelism.

    Column parallel (split output rows): qkv_proj, in_proj, gate_up_proj.
    Row parallel (split input cols): o_proj, out_proj, down_proj.
    Shared-expert and attention projections follow Megatron TP conventions.
    """
    if world == 1:
        return t
    # 128x128 2D block-fp8 has float32 2D scale matrices where row slicing would cut across multi-row scale blocks.
    # MXFP8 has uint8 1D per-row scales [N, K//32], so each row is completely independent.
    if name.endswith(".weight_scale_inv") and t.dtype != torch.uint8:
        raise NotImplementedError("qwen4_exp tensor parallelism serves bf16 or MXFP8 dense weights, not 128x128 block-fp8 ones")
    if name.endswith(".in_proj.weight") and t.dtype in _QUANT_DTYPES:
        raise NotImplementedError("qwen4_exp tensor parallelism serves bf16 or split MXFP8 in_proj, not unsplit block-fp8 in_proj")
    if name.endswith((".self_attn.qkv_proj.weight", ".self_attn.qkv_proj.weight_scale_inv")):
        q = (config.num_qo_heads, 2 * config.head_dim)
        kv = (config.num_kv_heads, config.head_dim)
        return _shard_rows(t, [q, kv, kv], rank, world)
    if ".linear_attn." in name:
        g = config.linear_attention_group()
        k = (g.num_key_heads, g.key_head_dim)
        v = (g.num_value_heads, g.value_head_dim)
        if name.endswith((".in_proj.weight", ".in_proj.weight_scale_inv")):
            return _shard_rows(t, [k, k, v, v, (v[0], 1), (v[0], 1)], rank, world)
        if name.endswith((".in_proj_qkvz.weight", ".in_proj_qkvz.weight_scale_inv")):
            return _shard_rows(t, [k, k, v, v], rank, world)
        if name.endswith((".in_proj_ba.weight", ".in_proj_ba.weight_scale_inv")):
            return _shard_rows(t, [(v[0], 1), (v[0], 1)], rank, world)
        if name.endswith(".conv1d.weight"):
            return _shard_rows(t, [k, k, v], rank, world)
        if name.endswith((".A_log", ".dt_bias")):
            return _shard_rows(t, [(v[0], 1)], rank, world)
        if name.endswith((".out_proj.weight", ".out_proj.weight_scale_inv")):
            return t.chunk(world, dim=1)[rank].clone()
        return t
    if name.endswith((".shared_expert.gate_up_proj.weight", ".shared_expert.gate_up_proj.weight_scale_inv")):
        half = t.shape[0] // 2
        return _shard_rows(t, [(half, 1), (half, 1)], rank, world)
    # o_proj / down_proj: dim 1; embed_tokens / lm_head: vocab rows; others unchanged.
    if name.endswith((
        ".self_attn.o_proj.weight",
        ".self_attn.o_proj.weight_scale_inv",
        ".shared_expert.down_proj.weight",
        ".shared_expert.down_proj.weight_scale_inv",
    )):
        return t.chunk(world, dim=1)[rank].clone()
    if name.endswith(("model.embed_tokens.weight", "lm_head.weight")):
        return t.chunk(world, dim=0)[rank].clone()
    return t


def _shard_vision_tensor(name: str, tensor: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    """Shard a vision tower tensor across TP ranks to match parallel layer layout."""
    if world == 1:
        return tensor
    if name.endswith(".attn.qkv.weight"):
        total_dim = tensor.shape[0]
        per_rank_qkv = total_dim // (3 * world)
        qkv_chunks = tensor.chunk(3, dim=0)
        return torch.cat([c.chunk(world, dim=0)[rank] for c in qkv_chunks], dim=0).clone()
    if name.endswith(".attn.proj.weight"):
        return tensor.chunk(world, dim=1)[rank].clone()
    if name.endswith(".mlp.linear_fc1.weight"):
        return tensor.chunk(world, dim=0)[rank].clone()
    if name.endswith(".mlp.linear_fc2.weight"):
        return tensor.chunk(world, dim=1)[rank].clone()
    if name.endswith(".mlp.linear_fc2.bias"):
        return tensor if rank == 0 else torch.zeros_like(tensor)
    return tensor


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
    Each Linear is read under the scheme the checkpoint's QuantConfig gives it, the same answer the model built its buffers from, so bf16, per-channel / per-tensor fp8, block-fp8, MXFP8 and NVFP4 projections in any mix land as the model's state dict.
    Fusions, per role: attention q|k|v -> ``qkv_proj``; GDN ``in_proj_{qkv,z,b,a}`` -> ``in_proj``, or ``in_proj_qkvz`` + ``in_proj_ba`` when qkv|z is quantized; shared-expert gate|up -> ``gate_up_proj``; each per-layer HC's ``input_mix_weight_down`` | ``block_inject_weight`` -> a zero-padded ``input_mix_weight_down_block_inject``.
    ``include_moe_experts`` is accepted for the loader contract but never yields anything: the routed experts are NVFP4 and always come from the offload cache's expert reader.
    """
    if not include_non_moe:
        return

    from .config import parse_config

    tp = get_tp_info()
    hf_config = cached_load_hf_config(model_path)
    spec = get_model_spec(hf_config.architectures[0])
    config = parse_config(hf_config) if tp.size > 1 else None
    reader = _DenseReader(get_quant_config(), spec.packed_modules_mapping)

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
                if name.startswith(VISION_KEY_PREFIXES):
                    yield name, _shard_vision_tensor(name, tensor, tp.rank, tp.size)
                    continue
                emitted = reader.add(name, tensor)
                if emitted is None:
                    yield name, _shard(name, tensor, config, tp.rank, tp.size)
                else:
                    for fused_name, fused_tensor in emitted:
                        yield fused_name, _shard(fused_name, fused_tensor, config, tp.rank, tp.size)

    if reader.pending:
        lines = reader.missing()
        shown = "\n  ".join(lines[:8]) + (f"\n  ... {len(lines) - 8} more" if len(lines) > 8 else "")
        raise ValueError(f"checkpoint is missing tensors the quant config declares for {len(lines)} modules:\n  {shown}")


# routed experts: expert_pieces.iter_extra_layer_expert_pieces
_MTP_EXPERTS_PREFIX = "mtp.layers.0.mlp.experts."
_MTP_EXPERT_RE = re.compile(re.escape(_MTP_EXPERTS_PREFIX) + r"(?:(?P<expert>\d+)\.(?P<proj>gate|up|down)_proj\.(?P<kind>weight|weight_scale_inv)|(?P<stacked>gate_up|down)_proj)$")


def iter_mtp_weights(model_path: str, device: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    """The ``mtp.*`` head but its routed experts, named as Qwen4ExpMTPDraft names it and fused like the target's dense weights."""
    hf_config = cached_load_hf_config(model_path)
    reader = _DenseReader(get_quant_config(), get_model_spec(hf_config.architectures[0]).packed_modules_mapping)
    seen = False
    for file in iter_weight_files(model_path):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                if not raw_name.startswith("mtp.") or raw_name.endswith((".k_scale", ".v_scale", ".q_scale", ".prob_scale")):
                    continue
                seen = True
                if raw_name.startswith(_MTP_EXPERTS_PREFIX):
                    continue
                tensor = f.get_tensor(raw_name)
                emitted = reader.add(raw_name, tensor)
                if emitted is None:
                    yield raw_name[len("mtp."):], tensor
                else:
                    yield from ((n.removeprefix("mtp."), t) for n, t in emitted)
    if not seen:
        raise ValueError(f"{model_path} has no mtp.* weights")
    if reader.pending:
        raise ValueError(f"{model_path}: incomplete MTP projections: {reader.missing()}")


def iter_mtp_expert_pieces(model_path: str):
    """The MTP head's routed experts: the stacked bf16 pair, or per-expert projections with their block-fp8 scales."""
    from freetoken.moe.expert_pieces import iter_extra_layer_expert_pieces

    def locate(name: str):
        m = _MTP_EXPERT_RE.match(name)
        if m is None:
            return None
        if m["stacked"]:
            return None, m["stacked"]
        return int(m["expert"]), m["proj"] + ("_scale" if m["kind"] == "weight_scale_inv" else "")

    return iter_extra_layer_expert_pieces(model_path, _MTP_EXPERTS_PREFIX, locate)


def iter_vision_weights(model_path: str, device: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    """The vision tower alone, named as iter_weights names it."""
    tp = get_tp_info()
    for file in iter_weight_files(model_path):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _rename(raw_name)
                if name is not None and name.startswith(VISION_KEY_PREFIXES):
                    yield name, _shard_vision_tensor(name, f.get_tensor(raw_name), tp.rank, tp.size)


# ======================================================================================
# PLE n-gram table
# ======================================================================================


@dataclass(frozen=True)
class PleTable:
    """The filled n-gram table: one pinned host bank plus the checkpoint's per-tensor scale (1 for a bf16 table).

    FP8 checkpoints fill the bank with e4m3 rows; NVFP4 checkpoints fill it with packed
    uint8 rows (two e2m1 codes per byte) plus ``scale_bank``, the per-16-element fp8 block
    scales, and keep the global scalar in ``weight_scale`` (``weight_scale_2`` upstream)."""

    bank: HostBank
    weight_scale: torch.Tensor  # scalar, checkpoint dtype (bf16)
    scale_bank: HostBank | None = None  # ``[total_rows, ngram_head_dim // 16]`` fp8, NVFP4 only

    @property
    def tensor(self) -> torch.Tensor:
        """``[total_rows, ngram_head_dim]`` view of the bank in the stored dtype."""
        return self.bank.tensor


_PLE_ST_DTYPES = {"F8_E4M3": torch.float8_e4m3fn, "BF16": torch.bfloat16, "U8": torch.uint8}
_PLE_SCALE2_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight_scale_2"
_PLE_SHARD_SCALE_RE = re.compile(
    r"\.ple\.ple_embedding\.ngram_embedding\.shard_(?P<shard>\d+)\.weight_scale$"
)


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



_PLE_FILE_BYTES = 4 << 30  # ple-table-*.safetensors written by ftw_side_files


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

def dump_ple_table(folder: str, shards: dict[str, torch.Tensor],
                   qwen4_args, *, workers: int = 4) -> list[str]:
    """Write the n-gram table tensors into ~10 GiB shards like the real checkpoint does."""
    import safetensors.torch
    _PLE_FILE_BYTES = 10 << 30
    written = []
    batch: dict[str, torch.Tensor] = {}
    size = 0
    file_idx = 0

    def flush():
        nonlocal batch, size, file_idx
        if not batch:
            return
        out_path = os.path.join(folder, f"model-plefp8-{file_idx:05d}.safetensors")
        safetensors.torch.save_file(batch, out_path)
        written.append(out_path)
        batch = {}
        size = 0
        file_idx += 1

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
    table is ~47.7 GiB (fp8) and must not also sit in the page cache while the bank holds the same bytes.
    An fp8 table needs the checkpoint's scalar ``weight_scale``; a bf16 table without one is unscaled.
    """
    folder = download_hf_weight(model_path)
    parts: dict[int, tuple[str, int, int]] = {}  # shard index -> (path, file offset, bytes)
    scale_parts: dict[int, tuple[str, int, int]] = {}
    scale: torch.Tensor | None = None
    scale2: torch.Tensor | None = None
    rows = cols = 0
    dtype: torch.dtype | None = None
    packed = False

    for path in _ple_table_files(folder):
        header, base = _safetensors_header(path)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if key.endswith(_PLE_SCALE2_SUFFIX):
                with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                    scale2 = f.get_tensor(key).reshape(())
                continue
            if key.endswith(_PLE_SCALE_SUFFIX):
                with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                    scale = f.get_tensor(key).reshape(())
                continue
            match = _PLE_SHARD_RE.search(key)
            if match is None:
                match = _PLE_SHARD_SCALE_RE.search(key)
                if match is not None:
                    if meta["dtype"] != "F8_E4M3":
                        raise ValueError(f"PLE scale shard {key} has unsupported dtype {meta['dtype']}")
                    begin, end = meta["data_offsets"]
                    scale_parts[int(match.group("shard"))] = (path, base + begin, end - begin)
                continue
            shard_dtype = _PLE_ST_DTYPES.get(meta["dtype"])
            if shard_dtype is None or dtype not in (None, shard_dtype):
                raise ValueError(f"PLE table shard {key} has dtype {meta['dtype']}, expected one of {sorted(_PLE_ST_DTYPES)} for every shard")
            dtype = shard_dtype
            if meta["dtype"] == "U8":
                packed = True
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
    if packed and sorted(scale_parts) != list(range(expected)):
        raise ValueError(
            f"NVFP4 PLE needs scale shards 0..{expected - 1}, found {len(scale_parts)}"
        )
    expected_dim = cols * (2 if packed else 1)
    if expected_dim != qwen4_args.ngram_head_dim:
        raise ValueError(f"PLE table row is {expected_dim} wide, config says {qwen4_args.ngram_head_dim}")

    scalar = scale2 if packed else scale
    if scalar is None:
        if dtype is torch.float8_e4m3fn:
            raise ValueError("PLE table has no weight_scale")
        scalar = torch.tensor(1.0, dtype=torch.bfloat16)

    tp = get_tp_info()
    use_shm = tp.size > 1 and os.path.exists("/dev/shm")
    shm_w = f"/dev/shm/ft_ple_weight_{rows}_{cols}.bin" if use_shm else None
    shm_s = f"/dev/shm/ft_ple_scale_{rows}_{cols}.bin" if use_shm and packed else None
    flag = f"/dev/shm/ft_ple_ready_{rows}_{cols}.flag" if use_shm else None
    is_primary = tp.rank == 0 if use_shm else True

    if use_shm and is_primary and os.path.exists(flag):
        try:
            os.remove(flag)
        except OSError:
            pass

    bank = HostBank((expected * rows, cols), dtype, shm_path=shm_w)
    scale_bank = HostBank((expected * rows, cols // 8), torch.float8_e4m3fn, shm_path=shm_s) if packed else None
    shard_bytes = rows * cols * dtype.itemsize
    scale_bytes = rows * (cols // 8) if packed else 0

    if is_primary:
        bar = byte_bar(expected * (shard_bytes + scale_bytes), "Loading PLE table")
        jobs = [(parts, bank, shard_bytes)]
        if packed and scale_bank is not None:
            jobs.append((scale_parts, scale_bank, scale_bytes))
        try:
            for src, dst, nbytes in jobs:
                buf = dst.memoryview()
                for shard in range(expected):
                    path, offset, n = src[shard]
                    assert n == nbytes, f"PLE shard {shard} is {n} B, expected {nbytes}"
                    read_range_into(buf, path, file_offset=offset, nbytes=n,
                                    dest_offset=shard * nbytes, workers=workers, chunk=chunk)
                    bar.update(n)
        finally:
            bar.close()
        if use_shm:
            with open(flag, "w") as f:
                f.write("ready")
    else:
        import time
        while not os.path.exists(flag):
            time.sleep(0.5)

    if pin and torch.cuda.is_available():
        bank.pin()
        if scale_bank is not None:
            scale_bank.pin()
    return PleTable(bank=bank, weight_scale=scalar, scale_bank=scale_bank)


# ======================================================================================


def nvfp4_expert_spec(model_path: str, config) -> Nvfp4ExpertSourceSpec:
    """The per-expert NVFP4 layout under the checkpoint's dialect names (ModelOpt or llm-compressor)."""
    quant = get_quant_config()
    stored = quant.stored_tensors(QuantKind.NVFP4)
    kind_map = {stored[role].name: kind for role, kind in _BANK_KINDS.items()}
    return Nvfp4ExpertSourceSpec(
        key_pattern=re.compile(_EXPERT_KEY_RE.format(kinds="|".join(map(re.escape, kind_map)))),
        proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
        layer_to_bank=lambda layer, config: layer,  # every layer is MoE
        desc=f"Qwen3.8-Flash-Next NVFP4 experts ({quant.dialect})",
        kind_map=kind_map,
        global_reciprocal=stored["weight_global"].reciprocal,
    )


__all__ = [
    "ftw_side_files",
    "PleTable",
    "iter_mtp_expert_pieces",
    "iter_mtp_weights",
    "iter_vision_weights",
    "iter_weights",
    "load_ple_table",
    "nvfp4_expert_spec",
]
