#include <iostream>
#include <mpi.h>
#include <cuda_runtime.h>

__global__ void writeData(int* data, int rank, int world_size) {
    if (threadIdx.x < world_size) {
        data[rank] = rank;  // Write rank to its own position
        printf("Rank %d writing to position %d\n", rank, rank);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Initialize CUDA
    cudaSetDevice(rank);

    // Allocate memory for all ranks on each device
    int* data;
    cudaMalloc((void**)&data, world_size * sizeof(int));
    cudaIpcMemHandle_t handle;
    cudaIpcGetMemHandle(&handle, data);

    // Gather all handles
    cudaIpcMemHandle_t* handles = new cudaIpcMemHandle_t[world_size];
    MPI_Allgather(&handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, handles, sizeof(cudaIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD);

    // Write rank to each device's memory
    for (int i = 0; i < world_size; i++) {
        if (i != rank) { // Skip own memory
            int* remoteData;
            cudaIpcOpenMemHandle((void**)&remoteData, handles[i], cudaIpcMemLazyEnablePeerAccess);
            writeData<<<1, world_size>>>(remoteData, rank, world_size);
            cudaDeviceSynchronize();
            cudaIpcCloseMemHandle(remoteData);
        }
    }

    // Cleanup
    cudaFree(data);
    delete[] handles;

    MPI_Finalize();
    return 0;
}