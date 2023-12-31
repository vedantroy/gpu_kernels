#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

#include <limits>
#include <vector>

#include "cuda_profiler_api.h"
#include "fast_allreduce.cuh"
#include "mpi.h"
#include "nccl.h"

#define MPICHECK(cmd)                                                  \
  do {                                                                 \
    int e = cmd;                                                       \
    if (e != MPI_SUCCESS) {                                            \
      printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

#define NCCLCHECK(cmd)                                              \
  do {                                                              \
    ncclResult_t r = cmd;                                           \
    if (r != ncclSuccess) {                                         \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, \
             ncclGetErrorString(r));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

__global__ void dummy_kernel() {
  for (int i = 0; i < 100; i++) __nanosleep(1000000);  // 100ms
}

template <typename T>
__global__ void set_data(T *data, int size, int myRank) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    data[idx] = myRank * 0.11f;
  }
}

template <typename T>
__global__ void convert_data(const T *data1, const T *data2, double *fdata1,
                             double *fdata2, int size) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    fdata1[idx] = data1[idx];
    fdata2[idx] = data2[idx];
  }
}

__global__ void init_rand(curandState_t *state, int size, int nRanks) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    for (int i = 0; i < nRanks; i++) {
      curand_init(i + 1, idx, 0, &state[idx * nRanks + i]);
    }
  }
}

template <typename T>
__global__ void gen_data(curandState_t *state, T *data, double *ground_truth,
                         int myRank, int nRanks, int size) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    double sum = 0.0;
    for (int i = 0; i < nRanks; i++) {
      double val = curand_uniform_double(&state[idx * nRanks + i]) * 4;
      T hval = val;  // downcast first
      sum += static_cast<double>(hval);
      if (i == myRank) data[idx] = hval;
    }
    ground_truth[idx] = sum;
  }
}

// T is half/etc.
template <typename T>
void run(int myRank, int nRanks, ncclComm_t &comm, int threads, int block_limit,
         int data_size) {
  T *result;
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDACHECK(cudaMalloc(&result, data_size * sizeof(T)));
  CUDACHECK(cudaMemset(result, 0, data_size * sizeof(T)));

  cudaIpcMemHandle_t self_data_handle;
  cudaIpcMemHandle_t data_handles[8];

  // 256 bytes => signal + counter (128 byte aligned)
  vllm::Metadata *buffer;
  T *self_data_copy;
  // Allocate 2 * data_size + space for vllm:Metadata
  CUDACHECK(
      cudaMalloc(&buffer, 2 * data_size * sizeof(T) + sizeof(vllm::Metadata)));
  CUDACHECK(cudaMemset(buffer, 0,
                       2 * data_size * sizeof(T) + sizeof(vllm::Metadata)));
  CUDACHECK(cudaMalloc(&self_data_copy, data_size * sizeof(T)));

  // Create a handle to the start of the buffer
  CUDACHECK(cudaIpcGetMemHandle(&self_data_handle, buffer));

  // MPI_Allgather(
  //   void* send_data,
  //   int send_count,
  //   MPI_Datatype send_datatype,
  //   void* recv_data,
  //   int recv_count,
  //   MPI_Datatype recv_datatype,
  //   MPI_Comm communicator)

  MPICHECK(MPI_Allgather(&self_data_handle, sizeof(cudaIpcMemHandle_t),
                         MPI_BYTE, data_handles, sizeof(cudaIpcMemHandle_t),
                         MPI_BYTE, MPI_COMM_WORLD));

  void *rank_data;
  size_t rank_data_sz = 16 * 1024 * 1024;
  CUDACHECK(cudaMalloc(&rank_data, rank_data_sz));
  std::vector<int64_t> offsets(nRanks, 0);
  vllm::FastAllreduce fa(buffer, rank_data, rank_data_sz, data_handles, offsets,
                         myRank);
  auto *self_data =
      reinterpret_cast<T *>(reinterpret_cast<char *>(buffer) +
                            sizeof(vllm::Metadata) + data_size * sizeof(T));
  // hack buffer registration
  {
    std::vector<std::string> handles;
    handles.reserve(nRanks);
    for (int i = 0; i < nRanks; i++) {
      char *begin = (char *)&data_handles[i];
      char *end = (char *)&data_handles[i + 1];
      // handles.emplace_back(begin, end);
      handles.emplace_back(begin, end);
    }
    std::vector<int64_t> offsets(
        nRanks, sizeof(vllm::Metadata) + data_size * sizeof(T));
    fa.register_buffer(handles, offsets, self_data);
  }

  // Ground truth data
  double *verification_buffer;
  CUDACHECK(cudaMallocHost(&verification_buffer, data_size * sizeof(double)));

  curandState_t *states;
  CUDACHECK(cudaMalloc(&states, sizeof(curandState_t) * nRanks * data_size));

  // blocks, threads per block
  init_rand<<<108, 1024, 0, stream>>>(states, data_size, nRanks);

  // self_data is inputs for current rank
  gen_data<T><<<108, 1024, 0, stream>>>(states, self_data, verification_buffer,
                                        myRank, nRanks, data_size);

  CUDACHECK(cudaMemcpyAsync(self_data_copy, self_data, data_size * sizeof(T),
                            cudaMemcpyDeviceToDevice, stream));
  cudaEvent_t start, stop;
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&stop));

  ncclDataType_t ncclDtype;
  if (std::is_same<T, half>::value) {
    ncclDtype = ncclFloat16;
  } else if (std::is_same<T, nv_bfloat16>::value) {
    ncclDtype = ncclBfloat16;
  } else {
    ncclDtype = ncclFloat;
  }

  dummy_kernel<<<1, 1, 0, stream>>>();
  constexpr int warmup_iters = 5;
  constexpr int num_iters = 25;
  // warmup
  for (int i = 0; i < warmup_iters; i++) {
    NCCLCHECK(ncclAllReduce(result, result, data_size, ncclDtype, ncclSum, comm,
                            stream));
  }
  CUDACHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < num_iters; i++) {
    NCCLCHECK(ncclAllReduce(result, result, data_size, ncclDtype, ncclSum, comm,
                            stream));
  }
  CUDACHECK(cudaEventRecord(stop, stream));
  CUDACHECK(cudaStreamSynchronize(stream));
  float allreduce_ms = 0;
  cudaEventElapsedTime(&allreduce_ms, start, stop);

  // if (myRank == 1) dummy_kernel<<<1, 1, 0, stream>>>();
  // set_data<T><<<16, 1024, 0, stream>>>(self_data, data_size, myRank);

  dummy_kernel<<<1, 1, 0, stream>>>();
  // warm up
  for (int i = 0; i < warmup_iters; i++) {
    // Method sig:
    // cudaStream_t, half *, half *, int, int, int
    fa.allreduce<T>(stream, self_data, result, data_size, threads, block_limit);
  }
  CUDACHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < num_iters; i++) {
    fa.allreduce<T>(stream, self_data, result, data_size, threads, block_limit);
  }
  CUDACHECK(cudaEventRecord(stop, stream));
  CUDACHECK(cudaStreamSynchronize(stream));

  float duration_ms = 0;
  cudaEventElapsedTime(&duration_ms, start, stop);
  if (myRank == 0)
    printf(
        "Rank %d done, nGPUs:%d, sz (kb): %d, %d, %d, my time:%.2fus, nccl "
        "time:%.2fus\n",
        myRank, nRanks, data_size * sizeof(T) / 1024, threads, block_limit,
        duration_ms * 1e3 / num_iters, allreduce_ms * 1e3 / num_iters);

  // And wait for all the queued up work to complete
  CUDACHECK(cudaStreamSynchronize(stream));

  // nccl result is in self_data
  NCCLCHECK(ncclAllReduce(self_data_copy, self_data, data_size, ncclDtype,
                          ncclSum, comm, stream));

  double *nccl_result, *my_result;
  CUDACHECK(cudaMallocHost(&nccl_result, data_size * sizeof(double)));
  CUDACHECK(cudaMallocHost(&my_result, data_size * sizeof(double)));

  // copy self_data -> nccl_result
  // copy result -> my_result
  convert_data<T><<<108, 1024, 0, stream>>>(self_data, result, nccl_result,
                                            my_result, data_size);
  CUDACHECK(cudaStreamSynchronize(stream));

  for (unsigned long j = 0; j < data_size; j++) {
    auto diff = abs(nccl_result[j] - my_result[j]);
    if (diff >= 1e-2) {
      printf("Rank %d: Verification mismatch at %lld: %f != (my) %f\n", myRank,
             j, nccl_result[j], my_result[j]);
      break;
    }
  }

  long double nccl_diffs = 0.0;
  long double my_diffs = 0.0;
  for (int j = 0; j < data_size; j++) {
    nccl_diffs += abs(nccl_result[j] - verification_buffer[j]);
    my_diffs += abs(my_result[j] - verification_buffer[j]);
  }
  if (myRank == 0)
    std::cout << "average abs diffs: nccl: " << nccl_diffs / data_size
              << " me: " << my_diffs / data_size << std::endl;

  CUDACHECK(cudaFree(result));
  CUDACHECK(cudaFree(self_data_copy));
  CUDACHECK(cudaFree(rank_data));
  CUDACHECK(cudaFree(buffer));
  CUDACHECK(cudaFree(states));
  CUDACHECK(cudaFreeHost(verification_buffer));
  CUDACHECK(cudaFreeHost(nccl_result));
  CUDACHECK(cudaFreeHost(my_result));
  CUDACHECK(cudaStreamDestroy(stream));
}

int main(int argc, char **argv) {
  int nRanks, myRank;
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));
  CUDACHECK(cudaSetDevice(myRank));
  ncclUniqueId id;
  ncclComm_t comm;
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast(static_cast<void *>(&id), sizeof(id), MPI_BYTE, 0,
                     MPI_COMM_WORLD));
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

  cudaProfilerStart();
  // for (int threads : {256, 512}) {
  //   for (int block_limit = 16; block_limit < 112; block_limit += 4) {
  //     run<half>(myRank, nRanks, comm, threads, block_limit, 4096 * 1024);
  //   }
  // }
  for (int sz = 512; sz <= (4 << 20); sz *= 2) {
    run<half>(myRank, nRanks, comm, 512, 36, sz + 8 * 50);
  }

  cudaProfilerStop();
  MPI_Finalize(); // prevents MPI error
  return EXIT_SUCCESS;
}