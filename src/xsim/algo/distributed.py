"""Process-group facade keeping all DP/DDP mechanics in one place.

Genesis binds one GPU per process, so multi-GPU training means one torchrun
rank per GPU, each owning its own env stack and a DDP replica. This facade
handles the rank bookkeeping: GPU masking, nccl setup, model wrapping, and
the cross-rank reductions a collect/train loop needs. Without torchrun every
method degrades to a single-process no-op, so callers never branch.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch import nn


class Distributed:
    """Under torchrun (RANK/WORLD_SIZE env set) each rank pins its GPU — this
    must construct BEFORE gs.init, since Genesis reads the process's CUDA
    device — joins the nccl group, and wraps models in DDP."""

    def __init__(self):
        self.rank = int(os.environ.get("RANK", 0))
        self.world = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.enabled = self.world > 1
        if self.enabled:
            # mask, don't set_device: Genesis's quadrants runtime allocates on
            # physical GPU 0 no matter the current torch device, so each rank
            # must see exactly one GPU (its LOCAL_RANK'th visible one)
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                visible.split(",")[self.local_rank] if visible
                else str(self.local_rank))
            dist.init_process_group("nccl")

    @property
    def main(self) -> bool:
        return self.rank == 0

    def wrap(self, module: nn.Module) -> nn.Module:
        """DDP-wrap for the training forward/backward; act()/EMA keep using the
        raw module, whose parameters DDP shares and keeps in sync."""
        if not self.enabled:
            return module
        # device 0 = this rank's only visible GPU (see __init__ masking)
        return nn.parallel.DistributedDataParallel(module, device_ids=[0])

    def mean(self, value: float) -> float:
        if not self.enabled:
            return value
        t = torch.tensor([value], dtype=torch.float64, device="cuda")
        dist.all_reduce(t)
        return float(t.item()) / self.world

    def min_int(self, value: int) -> int:
        """Ranks record different sample counts (episode lengths vary), so
        training must run the same number of optimizer steps everywhere or the
        gradient all-reduce deadlocks; callers truncate to this."""
        if not self.enabled:
            return value
        t = torch.tensor([value], dtype=torch.int64, device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return int(t.item())

    def obs_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Global mean/std over every rank's samples (stats live in checkpoint
        buffers, so they must match across replicas)."""
        n = torch.tensor([float(x.shape[0])], device=x.device)
        s, sq = x.sum(dim=0), (x * x).sum(dim=0)
        if self.enabled:
            for t in (n, s, sq):
                dist.all_reduce(t)
        mean = s / n
        var = (sq / n - mean * mean).clamp_min(0.0)
        return mean, var.sqrt()

    def close(self) -> None:
        if self.enabled:
            dist.barrier()
            dist.destroy_process_group()
