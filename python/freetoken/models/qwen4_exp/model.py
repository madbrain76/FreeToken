"""Qwen3.8-Flash-Next decoder stack (text-only).

The residual state is ``R [T, hc_count*hidden]`` end to end: the embedding is repeated over the
``hc_count`` streams, every layer mixes them down to one ``[T, hidden]`` block input and injects
its output back, and the top-level mixer collapses them once before ``lm_head``. There is no
input/post layernorm and no final ``model.norm`` -- the hyper-connection norms are the only ones.

Layer contract (frozen): ``forward(R [T, hc*hidden], batch) -> R' [T, hc*hidden]`` with an
immediate combine::

    R  = R + ple(R, batch)                 # zero-based layer 1 only
    x, s = attn_hc.mix(R); y = (GDN | QSA)(x); R = attn_hc.combine(R, y, s)
    x, s = mlp_hc.mix(R);  y = MoE(x);        R = mlp_hc.combine(R, y, s)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import Qwen4ExpAttention
from .config import PLE_NGRAM_STATE
from .hc import GatedResidual
from .moe import Qwen4ExpMoE
from .ple import PLELayer
from freetoken.models.blocks import embed_input_ids
from freetoken.models.qwen3_vl.vision import Qwen3VLVisionModel, QwenVLVisionMixin

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig


def _pop_chunk(chunks: list, index: int) -> torch.Tensor:
    chunk = chunks[index]
    chunks[index] = None
    return chunk


def _copy_layer_major_track_state(pool, layer, batch) -> None:
    """Seed a child at the one outer prefix-checkpoint boundary."""
    src = getattr(batch, "track_initial_src", None)
    if src is None:
        return
    dst = batch.track_initial_dst
    if layer._is_linear:
        index = pool.local_index(layer.linear_attn.layer_id)
        pool.recurrent_states[index, dst].copy_(pool.recurrent_states[index, src])
        pool.conv_states[index, dst].copy_(pool.conv_states[index, src])
    if layer.ple is not None:
        from .config import PLE_CONV_STATE, PLE_NGRAM_STATE

        pool.slot_state(PLE_CONV_STATE, layer.ple.layer_id)[dst].copy_(pool.slot_state(PLE_CONV_STATE, layer.ple.layer_id)[src])
        pool.slot_state(PLE_NGRAM_STATE)[dst].copy_(batch.ple_meta.ngram_context[0])


def build_linear_mixer(config: ModelConfig, layer_id: int, prefix: str) -> BaseOP:
    """GDN mixer of a linear_attention layer (Qwen3.5's GDN with a configurable output gate)."""
    from freetoken.layers.linear_attn import GatedDeltaNet

    g = config.linear_attention_group()
    return GatedDeltaNet(
        hidden_size=config.hidden_size,
        num_k_heads=g.num_key_heads,
        num_v_heads=g.num_value_heads,
        head_k_dim=g.key_head_dim,
        head_v_dim=g.value_head_dim,
        conv_kernel_size=g.conv_kernel_dim,
        rms_norm_eps=config.rms_norm_eps,
        layer_id=layer_id,
        output_gate=g.output_gate,
        quant_config=config.quant,
        prefix=prefix,
        out_heads_tiled=config.qwen4_args.gdn_out_heads_tiled,
    )


class Qwen4ExpDecoderLayer(BaseOP):
    """One decoder layer over the hyper-connection streams (see the module docstring for the flow)."""

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = "") -> None:
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            self.linear_attn = build_linear_mixer(config, layer_id, f"{prefix}.linear_attn")
        else:
            self.self_attn = Qwen4ExpAttention(config, layer_id, prefix=f"{prefix}.self_attn")
        self.mlp = Qwen4ExpMoE(config, layer_id, prefix=f"{prefix}.mlp")
        self.attn_hyper_connection = GatedResidual(config, prefix=f"{prefix}.attn_hyper_connection")
        self.mlp_hyper_connection = GatedResidual(config, prefix=f"{prefix}.mlp_hyper_connection")
        self.ple = (
            PLELayer(config, layer_id, prefix=f"{prefix}.ple") if layer_id in config.qwen4_args.ple_layer_ids else None
        )

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, batch: Batch, rn: torch.Tensor | None = None):
        """Returns (R, mlp_out, inject) un-combined: the caller fuses the mlp combine with the
        next consumer's norm (next layer's attn mix, or the final mixer)."""
        if self.ple is not None:
            hidden = hidden + self.ple.forward(hidden, batch)
        block_input, inject = self.attn_hyper_connection.mix(hidden, rn=rn)
        del rn
        if self._is_linear:
            block_output = self.linear_attn.forward(block_input)
        else:
            block_output = self.self_attn.forward(block_input, batch)
        del block_input
        hidden, rn_mlp = self.attn_hyper_connection.combine_norm(hidden, block_output, inject, self.mlp_hyper_connection)
        del block_output, inject
        block_input, inject = self.mlp_hyper_connection.mix(hidden, rn=rn_mlp)
        del rn_mlp
        return hidden, self.mlp.forward_parts(block_input), inject


class Qwen4ExpModel(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model") -> None:
        self.hc_count = config.qwen4_args.hc_count
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            quant_config=config.quant,
            prefix=f"{prefix}.embed_tokens",
        )
        self.layers = OPList(
            [
                Qwen4ExpDecoderLayer(config, layer_id, prefix=f"{prefix}.layers.{layer_id}")
                for layer_id in range(config.num_layers)
            ]
        )
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False, prefix=f"{prefix}.hyper_connection_mixer")
        # plain tuple (not an OP child), so it never shows up in the state dict
        self._ple = tuple(layer.ple for layer in self.layers.op_list if layer.ple is not None)

    @property
    def ple_layers(self) -> List[PLELayer]:
        """The PLE layers in decoder order -- the seam the loader attaches table backends to."""
        return list(self._ple)

    def forward(self, input_ids: torch.Tensor, batch: Batch) -> torch.Tensor:
        hidden = embed_input_ids(self.embed_tokens, input_ids, batch)
        hidden = hidden.repeat(1, self.hc_count)
        meta = None
        if self._ple:
            from .ple import build_ple_metadata, commit_ngram_context

            meta = build_ple_metadata(batch, self._ple[0].args, input_ids.device)
            for ple in self._ple:  # gather the pinned-host PLE rows while the early layers run
                ple.start_prefetch(batch, meta)
        # The last block of each layer combines through the layer boundary, fusing the
        # combine with the next consumer's hc_norm -- except before a PLE layer, whose
        #  happens after the boundary and changes what the norm sees.
        features = getattr(batch, "features", None)
        wanted = features.wanted_layers if features is not None else ()
        layers = self.layers.op_list
        rn = None
        for i, layer in enumerate(layers):
            hidden, mlp_out, inject = layer.forward(hidden, batch, rn=rn)
            nxt = layers[i + 1] if i + 1 < len(layers) else None
            if nxt is not None and nxt.ple is not None:
                hidden = layer.mlp_hyper_connection.combine(
                    hidden, layer.mlp_hyper_connection._epilogue(mlp_out), inject
                )
                rn = None
            else:
                consumer = nxt.attn_hyper_connection if nxt is not None else self.hyper_connection_mixer
                hidden, rn = layer.mlp_hyper_connection.combine_norm(hidden, mlp_out, inject, consumer)
            if i in wanted:
                features.layer_output(i, hidden, num_streams=self.hc_count)
        if meta is not None and meta.is_verify:
            # the commit rolls the context by the accepted rows only
            get_global_ctx().linear_state_pool.verify_handler(PLE_NGRAM_STATE).stash(meta)
        elif meta is not None:
            # single writer: the layers only read the context, so a second PLE layer's
            # prefetch sees the un-rolled window
            commit_ngram_context(meta, getattr(batch, "fla_metadata", None))
        if features is not None:
            # a draft reads the residual streams before the final mix
            features.final_hidden(hidden)
        return self.hyper_connection_mixer.mix(hidden, rn=rn)[0]

    def forward_layer_major(self, chunk_batches: List[Batch]) -> torch.Tensor:
        """Run decoder layers outermost, retaining the normal chunk order within each layer."""
        from .ple import build_ple_metadata, commit_ngram_context

        ctx = get_global_ctx()
        full_batch = ctx.batch
        features = getattr(full_batch, "features", None)
        wanted = features.wanted_layers if features is not None else ()
        hidden = []
        try:
            for child in chunk_batches:
                ctx._batch = child
                hidden.append(embed_input_ids(self.embed_tokens, child.input_ids, child).repeat(1, self.hc_count))

            for i, layer in enumerate(self.layers.op_list):
                for index, child in enumerate(chunk_batches):
                    ctx._batch = child
                    if layer.ple is not None:
                        child.ple_meta = build_ple_metadata(child, layer.ple.args, child.input_ids.device)
                    _copy_layer_major_track_state(ctx.linear_state_pool, layer, child)
                    h, mlp_out, inject = layer.forward(_pop_chunk(hidden, index), child)
                    hidden[index] = layer.mlp_hyper_connection.combine(
                        h, layer.mlp_hyper_connection._epilogue(mlp_out), inject
                    )
                    del h, mlp_out, inject
                    if layer.ple is not None:
                        commit_ngram_context(child.ple_meta, child.fla_metadata)
                if i in wanted:
                    features.layer_output(i, hidden, num_streams=self.hc_count)

            if features is not None:
                features.final_hidden(hidden)
            last = hidden.pop()
            hidden.clear()
            res = self.hyper_connection_mixer.mix(last)[0]
            del last
            return res
        finally:
            ctx._batch = full_batch


class Qwen4ExpForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self.model = Qwen4ExpModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            quant_config=config.quant,
            prefix="lm_head",
        )
        super().__init__()

    def load_host_tables(self, engine_config) -> int:
        """Attach the PLE n-gram table (pinned checkpoint bank, or zeros for dummy weights); returns the pinned host bytes the engine reserves from its pin budget."""
        ple_layers = self.model.ple_layers
        if not ple_layers:
            return 0
        from .ple import PinnedUVATable, ZeroTable, derive_ngram_hash_constants

        if getattr(engine_config, "use_dummy_weight", False):
            # Dummy fill leaves the int64 hash buffers garbage (a zero vocab size divides by
            # zero in the hash), so re-derive the real constants and read a zero table.
            for ple in ple_layers:
                args = ple.args
                mult, sizes, offsets = derive_ngram_hash_constants(
                    vocab_size=self._config.vocab_size,
                    ngram_size=args.ngram_size,
                    num_ngram_heads=args.num_ngram_heads,
                    ngram_vocab_size_base=args.ngram_vocab_size_base,
                    ple_layer_index=ple.ple_index,
                )
                emb = ple.ple_embedding
                emb.layer_multipliers.copy_(torch.tensor(mult, dtype=torch.int64))
                emb.ngram_heads_vocab_sizes.copy_(torch.tensor(sizes, dtype=torch.int64))
                emb.ngram_heads_offsets.copy_(torch.tensor(offsets, dtype=torch.int64))
                emb.attach_table(ZeroTable(offsets[-1] + sizes[-1], args.ngram_head_dim))
            return 0

        if engine_config.ple_backend == "disk":
            from .ple_disk import DiskRowTable, resolve_row_source
            from .weight import ple_table_is_packed

            if ple_table_is_packed(engine_config.model_path):
                raise NotImplementedError("NVFP4 PLE rows are only served by the pinned backend")

            # one WAIT node per captured graph: the flag protocol supports a single consume
            assert len(ple_layers) == 1, "disk PLE backend expects exactly one PLE layer"
            emb, args = ple_layers[0].ple_embedding, ple_layers[0].args
            # hash with the state-dict-loaded constants, the same source the pinned path reads
            constants = {
                "num_ngram_heads": args.num_ngram_heads,
                "layer_multipliers": emb.layer_multipliers.tolist(),
                "per_head_vocab_sizes": emb.ngram_heads_vocab_sizes.tolist(),
                "per_head_offsets": emb.ngram_heads_offsets.tolist(),
                "eos_token_id": args.ngram_boundary_token_id,
                "image_token_id": args.image_token_id,
            }
            rows = engine_config.decode_rows_per_req
            disk_table = DiskRowTable(
                resolve_row_source(engine_config.model_path),
                constants,
                # verify graphs stop at max_running_req requests, decode graphs at cuda_graph_max_bs
                max_graph_rows=engine_config.max_running_req * rows if rows > 1 else max(256, engine_config.cuda_graph_max_bs or 0),
                # an eager decode or verify step stages max_running_req * rows tokens, which can exceed a prefill chunk
                max_eager_tokens=max(engine_config.max_extend_tokens, engine_config.max_running_req * rows),
            )
            self._ple_table = disk_table
            for ple in ple_layers:
                ple.ple_embedding.attach_table(disk_table)
            # engine enters this around every dispatch; the graph itself never waits on the disk
            self.forward_host_ctx = disk_table.forward_host_ctx
            return 0

        from .weight import load_ple_table

        table = load_ple_table(engine_config.model_path, self._config.qwen4_args)
        self._ple_table = table  # owns the pinned HostBank; keep it alive
        packed = table.bank.tensor.dtype is torch.uint8
        uva = PinnedUVATable(
            table.bank.tensor,
            float(table.weight_scale),
            scales=table.scale_bank.tensor if packed else None,
        )
        for ple in ple_layers:
            ple.ple_embedding.attach_table(uva)
        return table.bank.nbytes + (table.scale_bank.nbytes if table.scale_bank is not None else 0)

    def forward(self) -> torch.Tensor:
        batch = get_global_ctx().batch
        # the model hands its pre-mix streams to batch.features itself
        return self.lm_head.forward(self.model.forward(batch.input_ids, batch), export_hidden=False)

    def forward_layer_major(self, chunk_batches: List[Batch]) -> torch.Tensor:
        ctx = get_global_ctx()
        previous_batch = ctx.batch
        try:
            hidden = self.model.forward_layer_major(chunk_batches)
            ctx._batch = chunk_batches[-1]
            out = self.lm_head.forward(hidden, export_hidden=False)
            del hidden
            return out
        finally:
            ctx._batch = previous_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class Qwen4ExpForConditionalGeneration(QwenVLVisionMixin, Qwen4ExpForCausalLM):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        if config.is_multimodal:
            assert not config.vision_config.deepstack_visual_indexes, "Qwen3.8 consumes no DeepStack features"
        if config.builds_vision_tower:
            self.visual = Qwen3VLVisionModel(config.vision_config, quant_config=config.quant, prefix="visual")


__all__ = [
    "Qwen4ExpDecoderLayer",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpModel",
    "build_linear_mixer",
]
