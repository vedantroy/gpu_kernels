#include "mpi.h"
#include "sync.cuh"

#define MPICHECK(cmd)                                                          \
  do {                                                                         \
    int e = cmd;                                                               \
    if (e != MPI_SUCCESS) {                                                    \
      printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e);         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

int main(int argc, char **argv) {
  MPI_Init(NULL, NULL);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  CUDACHECK(cudaSetDevice(world_rank));

  mysync::BarrierState *state;
  CUDACHECK(cudaMalloc(&state, sizeof(mysync::BarrierState)));

  cudaIpcMemHandle_t cur_rank_handle;
  cudaIpcMemHandle_t rank_handles[8];

  CUDACHECK(cudaIpcGetMemHandle(&cur_rank_handle, state));
  MPICHECK(MPI_Allgather(&cur_rank_handle,           // void* send_data,
                         sizeof(cudaIpcMemHandle_t), // int send_count,
                         MPI_BYTE,     // MPI_Datatype send_datatype,
                         rank_handles, // void* recv_data,
                         sizeof(cudaIpcMemHandle_t), // int recv_count,
                         MPI_BYTE,      // MPI_Datatype recv_datatype,
                         MPI_COMM_WORLD // MPI_Comm communicator
                         ));

  // Offsets are only necessary for Pytorch bindings
  // (where tensors are not allocated at the start of a cudaIpcMemHandle)
  // (that's why we set them to 0 here)
  std::vector<int64_t> offsets(world_size, 0);
  mysync::Sync sync(state, rank_handles, offsets, world_rank);

  sync.sync_test(8, 64);

  MPI_Finalize();
  return EXIT_SUCCESS;
}
