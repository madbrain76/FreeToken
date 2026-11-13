from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from freetoken.distributed import DistributedInfo
    from freetoken.kernel import PyNCCLCommunicator


@dataclass
class DistributedImpl(ABC):
    @abstractmethod
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def broadcast(self, x: torch.Tensor, src: int = 0) -> torch.Tensor: ...


@dataclass
class TorchDistributedImpl(DistributedImpl):
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        shape = list(x.shape)
        shape[0] = shape[0] * tp_size
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x)
        return out

    def broadcast(self, x: torch.Tensor, src: int = 0) -> torch.Tensor:
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return x
        backend = dist.get_backend()
        if x.is_cuda and backend == "gloo":
            cpu_t = x.cpu()
            dist.broadcast(cpu_t, src=src)
            x.copy_(cpu_t)
        else:
            dist.broadcast(x, src=src)
        return x



@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    comm: PyNCCLCommunicator

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        import tvm_ffi

        with tvm_ffi.use_torch_stream():
            self.comm.all_reduce(x, "sum")
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        import tvm_ffi
        from .info import get_tp_info

        world_size = get_tp_info().size
        output_shape = list(x.shape)
        output_shape[0] *= world_size
        result = x.new_empty(output_shape)
        with tvm_ffi.use_torch_stream():
            self.comm.all_gather(result, x)
        return result

    def broadcast(self, x: torch.Tensor, src: int = 0) -> torch.Tensor:
        import tvm_ffi
        from .info import get_tp_info

        world_size = get_tp_info().size
        if world_size <= 1:
            return x
        if x.is_cuda and x.nbytes > 0 and x.nbytes % 2 == 0:
            contig = x if x.is_contiguous() else x.contiguous()
            as_bf16 = contig.view(torch.uint8).view(-1).view(torch.bfloat16)
            gathered = as_bf16.new_empty((world_size * as_bf16.numel(),))
            with tvm_ffi.use_torch_stream():
                self.comm.all_gather(gathered, as_bf16)
            as_bf16.copy_(gathered.view(world_size, -1)[src])
            if contig is not x:
                x.copy_(contig)
            return x
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return x
        backend = dist.get_backend()
        if x.is_cuda and backend == "gloo":
            cpu_t = x.cpu()
            dist.broadcast(cpu_t, src=src)
            x.copy_(cpu_t)
        else:
            dist.broadcast(x, src=src)
        return x



class DistributedCommunicator:
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_gather(x)

    def broadcast(self, x: torch.Tensor, src: int = 0) -> torch.Tensor:
        return self.plugins[-1].broadcast(x, src=src)



def enable_pynccl_distributed(
    tp_info: DistributedInfo, tp_cpu_group: torch.distributed.ProcessGroup, max_bytes: int
) -> None:
    """
    Enable PyNCCL-based distributed communication for tensor parallelism.
    """
    if tp_info.size == 1:
        return
    from freetoken.kernel import init_pynccl

    comm = init_pynccl(
        tp_rank=tp_info.rank,
        tp_size=tp_info.size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_bytes,
    )

    DistributedCommunicator.plugins.append(PyNCCLDistributedImpl(comm))


def destroy_distributed() -> None:
    """
    Destroy all the distributed communication plugins.
    """
    DistributedCommunicator.plugins = []
