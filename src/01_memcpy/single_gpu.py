import cupy
import cupyx
import torch
from torch.cuda.nvtx import range_push, range_pop

N = 1024

if __name__ == "__main__":
    # =============== PyTorch =================
    a = torch.empty(N, N, N, device="cpu", pin_memory=True)
    b = torch.empty(N, N, N // 2, device=0)
    
    # This is pageable copy
    range_push("[PyTorch] copy from slice")
    b.copy_(a[:, :, :N // 2])
    range_pop()
    
    # contiguous() is necessary to make it pinned
    range_push("[PyTorch] copy from pinned slice")
    a_slice = torch.empty(N, N, N // 2, device="cpu", pin_memory=True)
    a_slice.copy_(a[:, :, :N // 2])
    b.copy_(a_slice)
    range_pop()
    
    # =============== CuPy =================
    a = cupyx.empty_pinned((N, N, N), dtype=cupy.float32)
    b = cupy.empty((N, N, N // 2), dtype=cupy.float32)
    
    # This is pageable copy
    # To make this work, CUPY_EXPERIMENTAL_SLICE_COPY=1 has to be set
    range_push("[CuPy] copy from slice")
    b[:] = a[:, :, :N // 2]
    range_pop()
    
    # copy to contiguous pinned memory
    range_push("[CuPy] copy from pinned slice")
    a_slice = cupyx.empty_pinned((N, N, N // 2), dtype=cupy.float32)
    a_slice[:] = a[:, :, :N // 2]
    b[:] = a_slice
    range_pop()
    
    