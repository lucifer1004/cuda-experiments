#include <chrono>
#include <cstring>
#include <iostream>

#include <common.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>

const int N = 1024;
GPUTimer timer;

__global__ void dummy_kernel(float *A) {
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t j = i; j < N * N * N; j += stride)
        A[j] = j;
}

void debug(size_t size, std::string extra_message = "",
           std::chrono::high_resolution_clock::time_point *start = nullptr) {
    float time;

    if (start == nullptr)
        time = timer.seconds();
    else {
        auto end = std::chrono::high_resolution_clock::now();
        time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - *start)
                .count() /
            1000000000.0;
    }
    if (extra_message != "")
        std::cout << "(" << extra_message << ") ";
    std::cout << "Time: " << time
              << " s; Bandwidth: " << (size) / (time * 1024 * 1024 * 1024)
              << " GB/s" << std::endl;
}

std::string get_flag(unsigned int flag) {
    std::string result;
    if (flag & cudaHostAllocPortable)
        result += "portable: true  |";
    else
        result += "portable: false |";

    if (flag & cudaHostAllocMapped)
        result += " mapped: true  |";
    else
        result += " mapped: false |";

    if (flag & cudaHostAllocWriteCombined)
        result += " write combined: true ";
    else
        result += " write combined: false";

    return result;
}

int main() {
    float *hA;

    std::cout << "==================================\n";
    std::cout << "malloc + memset\n\n";
    auto start = std::chrono::high_resolution_clock::now();
    hA = (float *)malloc(N * N * N * sizeof(float));
    memset(hA, 0, N * N * N * sizeof(float));
    debug(N * N * N * sizeof(float), "", &start);
    free(hA);

    std::cout << "==================================\n";
    std::cout << "calloc + memset\n\n";
    start = std::chrono::high_resolution_clock::now();
    hA = (float *)calloc(N * N * N, sizeof(float));
    memset(hA, 0, N * N * N * sizeof(float));
    debug(N * N * N * sizeof(float), "", &start);
    free(hA);

    float *dA;
    std::cout << "==================================\n";
    std::cout << "cudaHostAlloc + memset\n\n";
    for (unsigned int flag = 0; flag < 8; flag++) {
        timer.start();
        CHECK_CUDA_ERROR(
            cudaHostAlloc((void **)&hA, N * N * N * sizeof(float), flag));
        memset(hA, 0, N * N * N * sizeof(float));
        debug(N * N * N * sizeof(float), get_flag(flag));
        nvtxRangePush(get_flag(flag).c_str());
        if (flag & cudaHostAllocMapped) {
            timer.start();
            CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&dA, hA, 0));
            dummy_kernel<<<1024, 256>>>(dA);
            debug(N * N * N * sizeof(float), "used from device side");
        } else {
            timer.start();
            CHECK_CUDA_ERROR(cudaMalloc(&dA, N * N * N * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMemcpy(dA, hA, N * N * N * sizeof(float),
                                        cudaMemcpyHostToDevice));
            dummy_kernel<<<1024, 256>>>(dA);
            CHECK_CUDA_ERROR(cudaMemcpy(hA, dA, N * N * N * sizeof(float),
                                        cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaFree(dA));
            debug(N * N * N * sizeof(float), "used from device side");
        }
        nvtxRangePop();
        cudaFreeHost(hA);
    }

    std::cout << "==================================\n";
    std::cout << "cudaMalloc + cudaMemset\n\n";
    timer.start();
    CHECK_CUDA_ERROR(cudaMalloc(&dA, N * N * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(dA, 0, N * N * N * sizeof(float)));
    debug(N * N * N * sizeof(float));

    timer.start();
    dummy_kernel<<<1024, 256>>>(dA);
    debug(N * N * N * sizeof(float), "used from device side");
    CHECK_CUDA_ERROR(cudaFree(dA));

    std::cout << "==================================\n";

    return 0;
}
