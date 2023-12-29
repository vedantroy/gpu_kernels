#include <iostream>
#include <cuda_runtime.h>

__global__ void writeData(int* data, int rank, int world_size) {
    if (threadIdx.x < world_size) {
        data[rank] = rank;  // Write rank to its own position
        printf("Rank %d writing to position %d\n", rank, rank);
    }
}

extern "C" void gpu_work(int rank, int world_size) {
    // Initialize CUDA
    cudaSetDevice(rank);

    // Allocate memory and create IPC handle
    int* data;
    cudaMalloc((void**)&data, world_size * sizeof(int));
    cudaIpcMemHandle_t handle;
    cudaIpcGetMemHandle(&handle, data);

    // Gather all handles
    cudaIpcMemHandle_t* handles = new cudaIpcMemHandle_t[world_size];
    MPI_Allgather(&handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, handles, sizeof(cudaIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD);

    // Call the kernel for each device's memory
    for (int i = 0; i < world_size; i++) {
        if (i != rank) {
            int* remoteData;
            cudaIpcOpenMemHandle((void**)&remoteData, handles[i], cudaIpcMemLazyEnablePeerAccess);
            writeData<<<1, world_size>>>(remoteData, rank, world_size);
            cudaIpcCloseMemHandle(remoteData);
        }
    }

    // Cleanup
    cudaFree(data);
    delete[] handles;

    cudaDeviceSynchronize();
}