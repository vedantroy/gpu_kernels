#include <stdio.h>
#include <cuda_runtime.h>

__global__ void writeData(int* data, int rank, int world_size) {
    if (threadIdx.x < world_size) {
        data[rank] = rank;
        printf("Rank %d writing to position %d\n", rank, rank);
    }
}

void runWriteData(int* data, int rank, int world_size) {
    writeData<<<1, world_size>>>(data, rank, world_size);
}