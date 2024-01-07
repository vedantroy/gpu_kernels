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

// TODO: Why alignment of 16?
// TODO: Why even put this on the GPU at all?

// struct __align__(16) RankPtrs {
struct __align__(16) RankPtrs {
  const void *__restrict__ ptrs[8];
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

// like std::array, but aligned
// TODO: Exactly why does this matter?
template <typename T, int sz> struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

/*
Maximize memory efficiency w/ ld.128,st.128 instructions
The GPU cannot load more than 128 bytes at a time
CUDA threads can read a maximum of 16 bytes at once
So, each thread loads as many elements of type T as can be accommodated within
16 bytes
https://stackoverflow.com/questions/72147025/what-are-cuda-global-memory-32-64-and-128-byte-transactions
*/
template <typename T> struct packed_t {
  // the (P)acked type for load/store
  using P = array_t<T, 16 / sizeof(T)>;
  // the (A)ccumulator type for reduction
  using A = array_t<float, 16 / sizeof(T)>;
};

#define DINLINE __device__ __forceinline__

// scalar add functions
// for some reason when compiling with Pytorch, the + operator for half and
// bfloat is disabled so we call the intrinsics directly
DINLINE half &assign_add(half &a, half b) {
  a = __hadd(a, b);
  return a;
}
DINLINE float &assign_add(float &a, float b) { return a += b; }

template <typename T, int N>
DINLINE array_t<T, N> &packed_assign_add(array_t<T, N> &a, array_t<T, N> b) {
#pragma unroll
  for (int i = 0; i < N; i++) {
    assign_add(a.data[i], b.data[i]);
  }
  return a;
}

template <typename T, int N>
DINLINE array_t<float, N> upcast(array_t<T, N> val) {
  if constexpr (std::is_same<T, float>::value) {
    return val;
  } else {
    array_t<float, N> out;
#pragma unroll
    for (int i = 0; i < N; i++) {
      out.data[i] = upcast_s(val.data[i]);
    }
    return out;
  }
}

template <typename O> DINLINE O downcast(array_t<float, O::size> val) {
  if constexpr (std::is_same<typename O::type, float>::value) {
    return val;
  } else {
    O out;
#pragma unroll
    for (int i = 0; i < O::size; i++) {
      out.data[i] = downcast_s<typename O::type>(val.data[i]);
    }
    return out;
  }
}

template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P *ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < ngpus; i++) {
    packed_assign_add(tmp, upcast(ptrs[i][idx]));
  }
  return downcast<P>(tmp);
}

template <typename T, int ngpus>
__global__ void allreduce_kernel(
    RankPtrs buffer_ptrs, // Pointers to the buffer to reduce, 1 for each GPU
    RankSignals sg, volatile BarrierState *bstate, T *__restrict__ result,
    int rank, int world_size) {
  // Both P,A are array_t
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  const P *ptrs[ngpus];

  // TODO: Why are we loading the pointers in a circular fashion?
  // Since every allreduce sums across all ranks, we should be able to
  // load the pointers inorder
#pragma unroll
  for (int i = 0; i < world_size; i++) {
    int target = (rank + i) % world_size;
    ptrs[i] = (P *)buffer_ptrs->ptrs[target];
  }

  start_sync(sg, bstate, rank, world_size);

  // This is summing across all the ranks
  // at the given index
  // it's basically `result[idx] = rank1[idx] + rank2[idx] + rank3[idx]`
  // All complexity comes from
  // the packed type -- which means idx is actually a range of indices
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    // packed_reduce is just iterating over all the ranks
    ((P *)result)[idx] = packed_reduce<P, ngpus, A>(ptrs, idx);
  }

  end_sync(sg, bstate, rank, world_size);
}

class Sync {
public:
  int rank_;
  int world_size_;

  // Contains pointers to GPU memory
  RankSignals sg_;

  // Point to GPU memory
  RankPtrs *buffer_ptrs_;
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

  void register_buffer(const std::vector<std::string> &handles,
                       const std::vector<int64_t> &offsets, void *rank_ptr) {
    if (buffer_ptrs_ != nullptr) {
      throw std::runtime_error("register_buffer() called twice");
    }

    CUDACHECK(cudaMalloc(&buffer_ptrs_, sizeof(RankPtrs)));
    for (int i = 0; i < world_size_; i++) {
      if (i != rank_) {
        void *ptr;
        CUDACHECK(cudaIpcOpenMemHandle(
            &ptr, *((const cudaIpcMemHandle_t *)handles[i].data()),
            cudaIpcMemLazyEnablePeerAccess));
        ipc_handles_.push_back(ptr);
        ptr = (char *)ptr + offsets[i];
        (buffer_ptrs_)->ptrs[i] = ptr;
      } else {
        (buffer_ptrs_)->ptrs[i] = rank_ptr;
      }
    }
  }

  void sync_test(int blocks, int threads) {
    if (threads % 32 != 0 || threads <= 32) {
      throw std::runtime_error(
          "Threads must be a multiple of 32 greater than 32");
    }
    if (buffer_ptrs_ == nullptr) {
      throw std::runtime_error("register_buffer() must be called first");
    }
    switch (world_size_) {
    case 2:
      allreduce_kernel<half, 2><<<blocks, threads>>>(
          buffer_ptrs_, sg_, barrier_state_, rank_, world_size_);
      break;
    case 4:
      allreduce_kernel<half, 4><<<blocks, threads>>>(
          buffer_ptrs_, sg_, barrier_state_, rank_, world_size_);
      break;
    default:
      throw std::runtime_error("Unsupported world size");
    }
    cudaDeviceSynchronize();
  }

  ~Sync() {
    printf("Rank %d calling destructor\n", rank_);
    for (auto ptr : ipc_handles_) {
      CUDACHECK(cudaIpcCloseMemHandle(ptr));
    }
    /*
    if (buffer_ptrs_ != nullptr) {
      CUDACHECK(cudaFree(buffer_ptrs_));
    }
    */
  }
};
} // namespace mysync
