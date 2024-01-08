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
  // cudaMalloc once, and we can use the same cudaIpcMemHandle_t for all of them.
  
  auto *state_cpu = (mysync::BarrierState *)malloc(sizeof(mysync::BarrierState) + (2 * sizeof(DTYPE) * N_ELEMENTS));
  assert(state_cpu != NULL);

  DTYPE *input_buf_cpu = reinterpret_cast<DTYPE *>(state_cpu + 1);
  DTYPE *output_buf_cpu = input_buf_cpu + N_ELEMENTS;

  // set input_buf_cpu to all 1s
  for (int i = 0; i < N_ELEMENTS; ++i) {
    input_buf_cpu[i] = __float2half(1.0f);
  }

  // set output buf to all 0s
  // (not that this should matter)
  for (int i = 0; i < N_ELEMENTS; ++i) {
    output_buf_cpu[i] = __float2half(0.0f);
  }

  // copy the entire thing to the GPU
  mysync::BarrierState *state;
  CUDACHECK(cudaMalloc(&state, sizeof(mysync::BarrierState) + (2 * sizeof(DTYPE) * N_ELEMENTS)));
  CUDACHECK(cudaMemcpy(state, state_cpu, sizeof(mysync::BarrierState) + (2 * sizeof(DTYPE) * N_ELEMENTS), cudaMemcpyHostToDevice));

  DTYPE *input_buf = reinterpret_cast<DTYPE *>(state + 1);
  DTYPE *output_buf = input_buf + N_ELEMENTS;

  // mem copy to cpu and print first element
  DTYPE *input_buf_cpu2 = new DTYPE[N_ELEMENTS];
  CUDACHECK(cudaMemcpy(input_buf_cpu2, input_buf, N_ELEMENTS * sizeof(DTYPE), cudaMemcpyDeviceToHost));
  printf("Rank %d: input_buf[0] = %f\n", world_rank, __half2float(input_buf_cpu2[0]));

  DTYPE *output_buf_cpu2 = new DTYPE[N_ELEMENTS];
  CUDACHECK(cudaMemcpy(output_buf_cpu2, output_buf, N_ELEMENTS * sizeof(DTYPE), cudaMemcpyDeviceToHost));
  printf("Rank %d: output_buf[0] = %f\n", world_rank, __half2float(output_buf_cpu2[0]));

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
      char buffer[sizeof(cudaIpcMemHandle_t)];
      memcpy(buffer, &rank_handles[i], sizeof(cudaIpcMemHandle_t));
      handles[i] = std::string(buffer, sizeof(cudaIpcMemHandle_t));

      /*
      char *begin = (char *)(&rank_handles[i]);
      char *end1 = (char *)(&rank_handles[i + 1]);
      char *end2 = begin + sizeof(cudaIpcMemHandle_t);
      assert(end1 == end2);
      std::string handle_str(begin, end1);
      handles.push_back(handle_str);
      handles.emplace_back(begin, end1);
      handles.push_back(begin, end1);
      */
    }

    // {
    //   for (int i = 0; i < world_size; ++i) {
    //     if (i == world_rank) continue; // skip self (otherwise we get an 'invalid context' error)
    //     cudaIpcMemHandle_t handle = rank_handles[i];
    //     // printf("Rank %d: opening handle %d before registration\n", world_rank, i);
    //     char* ptr;
    //     CUDACHECK(cudaIpcOpenMemHandle((void **)&ptr, handle, cudaIpcMemLazyEnablePeerAccess));
    //     printf("Rank %d: opened handle %d before registration\n", world_rank, i);
    //   }
    // }

    // for (int i = 0; i < world_size; ++i) {
    //   if (i == world_rank) continue; // skip self (otherwise we get an 'invalid context' error)
    //   // cudaIpcMemHandle_t handle;
    //   // memcpy(&handle, handles[i].data(), sizeof(cudaIpcMemHandle_t));
    
    //   char* ptr;
    //   // CUDACHECK(cudaIpcOpenMemHandle((void **)&ptr, handle, cudaIpcMemLazyEnablePeerAccess));
    //   CUDACHECK(cudaIpcOpenMemHandle(
    //       (void**)&ptr, *((const cudaIpcMemHandle_t *)handles[i].data()),
    //       cudaIpcMemLazyEnablePeerAccess));
    //   printf("Rank %d: opened handle %d before registration (v2)\n", world_rank, i);
    // }

    std::vector<int64_t> buffer_offsets(world_size, sizeof(mysync::BarrierState));
    sync.register_buffer(handles, buffer_offsets, input_buf);
  }

  sync.sync_test<DTYPE>(N_ELEMENTS, output_buf);

  DTYPE *output_buf_cpu3 = new DTYPE[N_ELEMENTS];
  CUDACHECK(cudaMemcpy(output_buf_cpu3, output_buf, N_ELEMENTS * sizeof(DTYPE), cudaMemcpyDeviceToHost));
  printf("Rank %d: output_buf[0] = %f\n", world_rank, __half2float(output_buf_cpu3[0]));


  MPI_Finalize();
  return EXIT_SUCCESS;
}
