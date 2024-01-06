#include <cuda.h>
// #include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>
// #include <limits>
// #include <unordered_map>
#include <vector>

//#define CUDACHECK(cmd)                                              \
//  do {                                                              \
//    cudaError_t e = cmd;                                            \
//    if (e != cudaSuccess) {                                         \
//      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
//             cudaGetErrorString(e));                                \
//      exit(EXIT_FAILURE);                                           \
//    }                                                               \
//  } while (0)

#define CUDACHECK(cmd) cmd

__global__ void sync_test_kernel() {
  if (threadIdx.x == 0) {
    printf("Hello from block %d\n", blockIdx.x);
  }
}

namespace mysync {
struct Signal {
  alignas(64) union {
    uint64_t flag;
    unsigned char data[8];
  } start;
  alignas(64) union {
    uint64_t flag;
    unsigned char data[8];
  } end;
};

struct BarrierState {
  alignas(128) Signal sg;
  alignas(128) int counter;
};
static_assert(offsetof(BarrierState, counter) == 128);
static_assert(sizeof(BarrierState) == 256);

struct RankSignals {
  volatile Signal *signals[8];
};

class Sync {
    public:
      int rank_;
      int world_size_;

      // below are device pointers
      RankSignals sg_;
      BarrierState *barrier_state_;

      std::vector<void *> ipc_handles_;

      Sync(BarrierState *barrier_state, const cudaIpcMemHandle_t *handles,
           const std::vector<int64_t> &offsets, int rank)
           : rank_(rank),
             world_size_(offsets.size()),
             barrier_state_(barrier_state) {
        for (int i = 0; i < world_size_; i++) {
          BarrierState *rank_barrier_state;
          if (i != rank_) {
            char *handle;
            CUDACHECK(cudaIpcOpenMemHandle((void **)&handle, handles[i],
                                           cudaIpcMemLazyEnablePeerAccess));
            ipc_handles_.push_back(handle);
            handle += offsets[i];
            rank_barrier_state = (BarrierState *)handle;
          } else {
            rank_barrier_state = barrier_state_;
          }
          // This is pure pointer math (no access to on-device memory)
          sg_.signals[i] = &rank_barrier_state->sg;
        }
      }

      void sync_test(int blocks, int threads) {
        if (threads % 32 != 0 || threads <= 32) {
          throw std::runtime_error("Threads must be a multiple of 32 greater than 32");
        }
        sync_test_kernel<<<blocks, threads>>>();
      }

      ~Sync() {
          for (auto ptr : ipc_handles_) {
            CUDACHECK(cudaIpcCloseMemHandle(ptr));
          }
      }
};
}
