#include <cstdio>
#include <cstdlib>
#include <vector>

#include <common.h>
#include <mpi.h>
#include <nvToolsExt.h>

#define N (1 << 30)

int main() {
    int rank, ranks;
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    cudaSetDevice(rank);

    float *hA, *dA;

    std::string tag = "Rank " + std::to_string(rank);
    nvtxRangePush(tag.c_str());
    cudaHostAlloc(&hA, sizeof(float) * N, cudaHostAllocDefault);
    cudaMalloc(&dA, sizeof(float) * N);
    cudaMemcpy(dA, hA, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(hA, dA, sizeof(float) * N, cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFreeHost(hA);
    nvtxRangePop();
    MPI_Finalize();

    return 0;
}