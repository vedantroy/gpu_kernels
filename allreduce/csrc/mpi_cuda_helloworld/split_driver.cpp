#include <iostream>
#include <cuda_runtime.h>
#include <mpi.h>

extern void runWriteData(int* data, int rank, int world_size);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Initialize CUDA
    cudaSetDevice(rank);

    // Allocate memory and IPC handles
    int* data;
    cudaMalloc((void**)&data, world_size * sizeof(int));
    cudaIpcMemHandle_t handle;
    cudaIpcGetMemHandle(&handle, data);

    // Gather all handles
    cudaIpcMemHandle_t* handles = new cudaIpcMemHandle_t[world_size];
    MPI_Allgather(&handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, handles, sizeof(cudaIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD);

    // Write rank to each device's memory
    for (int i = 0; i < world_size; i++) {
        if (i != rank) {
            int* remoteData;
            cudaIpcOpenMemHandle((void**)&remoteData, handles[i], cudaIpcMemLazyEnablePeerAccess);
            runWriteData(remoteData, rank, world_size);
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
