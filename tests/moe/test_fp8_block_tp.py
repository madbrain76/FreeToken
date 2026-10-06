"""TP eligibility for replicated block-FP8 routed experts."""

from freetoken.layers.quantization.moe.base import MoEConfig
from freetoken.layers.quantization.moe.fp8_block import TritonFp8BlockMoEKernel


def test_block_fp8_kernel_accepts_replicated_tp_experts():
    cfg = MoEConfig(num_experts=512, hidden=2560, intermediate=640, top_k=10, tp_size=4)
    kernel = TritonFp8BlockMoEKernel()
    assert kernel.unusable_reason(cfg) is None
    assert [
        kernel.layout(MoEConfig(num_experts=512, hidden=2560, intermediate=640, top_k=10, tp_size=4, tp_rank=rank))["down"].shape[1]
        for rank in range(4)
    ] == [128, 128, 128, 256]
