import torch
import torch.distributed as dist
import os

from torch.cuda.nvtx import range_push, range_pop

N = 1024

if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    
    dist.init_process_group(backend="nccl")
    dist.barrier()
    range_push(f"Rank {rank}")
    A = torch.empty(N, N, N, device="cpu", dtype=torch.float32, pin_memory=True)
    B = torch.empty(N, N, N, device=rank, dtype=torch.float32)
    B.copy_(A)
    A.copy_(B)
    range_pop()
