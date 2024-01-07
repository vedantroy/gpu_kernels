#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <limits>
#include <vector>

#define CUDACHECK(cmd)                                                         \
  do {                                                                         \
    cudaError_t e = cmd;                                                       \
    if (e != cudaSuccess) {                                                    \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,            \
             cudaGetErrorString(e));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// #define CUDACHECK(cmd) cmd

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

__device__ uint64_t get_target_flag(int world_size) {
  // 64 1s
  auto m = std::numeric_limits<uint64_t>::max();
  // Each GPU gets 8 bits in the flag
  // E.g, if there are 4 GPUs, the target flag is
  // 32 0s followed by 32 1s
  return m >> ((8 - world_size) * 8);
}

__device__ void start_sync(const RankSignals &sg, volatile BarrierState *bstate,
                           int rank, int world_size) {
  bool first_block_in_rank = blockIdx.x == 0;
  if (first_block_in_rank) {
    if (threadIdx.x < world_size) {
      int other_rank = threadIdx.x;
      // warp 1: notify all other ranks that this rank has reached the sync
      // point
      sg.signals[other_rank]->start.data[rank] = 255;
    } else if (threadIdx.x == 32) {
      // warp 2: reset the end signal
      bstate->sg.end.flag = 0;
    }
  }

  // busy-wait until the current rank's signal
  // has been written to by all ranks
  if (threadIdx.x == 0) {
    uint64_t target_flag = get_target_flag(world_size);
    while (bstate->sg.start.flag != target_flag)
      ;
  }
  if (threadIdx.x == 0 && first_block_in_rank)
    printf("1st block rank %d done busy-wait\n", rank);
  __syncthreads();
}

__device__ void end_sync(const RankSignals &sg, volatile BarrierState *bstate,
                         int rank, int world_size) {
  __shared__ int blocks_at_sync_point;
  if (threadIdx.x == 0)
    blocks_at_sync_point = atomicAdd((int *)&bstate->counter, 1);
  __syncthreads(); // (I think) this ensures `blocks_at_sync_point` is assigned

  bool last_block_at_sync_point = (blocks_at_sync_point == gridDim.x - 1);
  if (last_block_at_sync_point) {
    if (threadIdx.x < world_size) {
      int other_rank = threadIdx.x;
      // warp 1: notify all other ranks that this rank has reached the sync
      // point
      sg.signals[other_rank]->end.data[rank] = 255;
    } else if (threadIdx.x == 32) {
      // warp 2: reset the start signal + counter
      bstate->sg.start.flag = 0;
      bstate->counter = 0;
    }
  }

  // busy-wait until the current rank's signal
  // has been written to by all ranks
  if (threadIdx.x == 0) {
    uint64_t target_flag = get_target_flag(world_size);
    while (bstate->sg.end.flag != target_flag)
      ;
  }
  __syncthreads();
}

#define NS_PER_S (uint64_t)1000000000

__global__ void sleepKernel() {
  uint64_t start, end;
  uint64_t sleepTime = 5 * NS_PER_S; // Sleep for 5 seconds

  if (threadIdx.x == 0) {
    // Record start time
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));

    // Sleep for 5 seconds
    __nanosleep(sleepTime);

    // Record end time
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(end));

    // Calculate and print the elapsed time in nanoseconds and milliseconds
    uint64_t elapsedNs = end - start;
    double elapsedMs = (double)elapsedNs / 1000000.0;
    printf("Slept for %llu nanoseconds (%.3f milliseconds)\n", elapsedNs,
           elapsedMs);
  }
}

// The %globaltimer register seems to not be working
__global__ void sync_test_kernel(RankSignals sg, volatile BarrierState *bstate,
                                 int rank, int world_size) {

  int sleep_time = (rank * NS_PER_S) + (blockIdx.x * NS_PER_S * 0.1);
  uint64_t start, end;
  if (threadIdx.x == 0) {
    printf("rank %d, block %d, sleep time: %d\n", rank, blockIdx.x, sleep_time);
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    __nanosleep((rank * NS_PER_S) + (blockIdx.x * NS_PER_S * 0.1));
  }
  __syncthreads();

  start_sync(sg, bstate, rank, world_size);

  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(end));
    printf("[start_sync] Hello from rank %d, block %d, elapsed time: %llu ns\n",
           rank, blockIdx.x, end - start);
    __nanosleep((rank * NS_PER_S) + (blockIdx.x * NS_PER_S * 0.1));
  }
  __syncthreads();

  end_sync(sg, bstate, rank, world_size);

  if (threadIdx.x == 0) {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(end));
    printf("[end_sync] Hello from rank %d, block %d, elapsed time: %llu ns\n",
           rank, blockIdx.x, end - start);
  }
  __syncthreads();
}

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
      : rank_(rank), world_size_(offsets.size()),
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
      throw std::runtime_error(
          "Threads must be a multiple of 32 greater than 32");
    }
    sleepKernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    sync_test_kernel<<<blocks, threads>>>(sg_, barrier_state_, rank_,
                                          world_size_);
    cudaDeviceSynchronize();
  }

  ~Sync() {
    printf("Rank %d calling destructor\n", rank_);
    for (auto ptr : ipc_handles_) {
      CUDACHECK(cudaIpcCloseMemHandle(ptr));
    }
  }
};
} // namespace mysync
