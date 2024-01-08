#include "allreduce.cuh"
#include "mpi.h"
#include <assert.h>

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

  #define N_ELEMENTS (1024 * 8)
  #define DTYPE half

  // We allocate the barrier state, input buffer, and output buffer, in one pass, so we only need to
  // create + allgather a single ipc handle
  mysync::BarrierState *state;
  CUDACHECK(cudaMalloc(&state, sizeof(mysync::BarrierState) + (2 * sizeof(DTYPE) * N_ELEMENTS)));

  DTYPE *input_buf = reinterpret_cast<DTYPE *>(state + 1);
  DTYPE *output_buf = input_buf + N_ELEMENTS;

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

  {
    // register the buffer
    std::vector<std::string> handles(world_size);
    handles.reserve(world_size);
    for (int i = 0; i < world_size; ++i) {
      char *begin = (char *)(&rank_handles[i]);
      char *end1 = (char *)(&rank_handles[i + 1]);
      char *end2 = begin + sizeof(cudaIpcMemHandle_t);
      assert(end1 == end2);
      handles.emplace_back(begin, end1);
    }
    {
      for (int i = 0; i < world_size; ++i) {
        if (i == world_rank) continue; // skip self (otherwise we get an 'invalid context' error)
        cudaIpcMemHandle_t handle = rank_handles[i];
        // printf("Rank %d: opening handle %d before registration\n", world_rank, i);
        char* ptr;
        CUDACHECK(cudaIpcOpenMemHandle((void **)&ptr, handle, cudaIpcMemLazyEnablePeerAccess));
        printf("Rank %d: opened handle %d before registration\n", world_rank, i);
      }
    }
    sync.register_buffer(handles, offsets, input_buf, rank_handles);
  }

  sync.sync_test<DTYPE>(N_ELEMENTS, output_buf);

  MPI_Finalize();
  return EXIT_SUCCESS;
}
