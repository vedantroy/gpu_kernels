#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "allreduce.cuh"

// we are returning a pointer as a uint64
// the static assert validates that pointers on this system
// are infact 64 bits
using fptr_t = uint64_t;
static_assert(sizeof(void *) == sizeof(fptr_t));

fptr_t init_ar(torch::Tensor &bstate, torch::Tensor &rank_data,
                    const std::vector<std::string> &handles,
                    const std::vector<int64_t> &offsets, int rank) {

  int world_size = offsets.size();
  if (world_size > 8)
    throw std::invalid_argument("world size > 8 is not supported");
  if (world_size % 2 != 0)
    throw std::invalid_argument("Odd num gpus is not supported for now");
  if (world_size != handles.size())
    throw std::invalid_argument(
        "handles length should equal to offsets length");
  if (rank < 0 || rank >= world_size)
    throw std::invalid_argument("invalid rank passed in");

  cudaIpcMemHandle_t ipc_handles[8];
  for (int i = 0; i < world_size; i++) {
    std::memcpy(&ipc_handles[i], handles[i].data(), sizeof(cudaIpcMemHandle_t));
  }
  return (fptr_t) new mysync::Sync(
      reinterpret_cast<mysync::BarrierState *>(bstate.data_ptr()), ipc_handles, offsets, rank);
}

void register_buffer(fptr_t _fa, torch::Tensor &t,
                     const std::vector<std::string> &handles,
                     const std::vector<int64_t> &offsets) {
  auto fa = reinterpret_cast<mysync::Sync *>(_fa);
  fa->register_buffer(handles, offsets, t.data_ptr());
}

void allreduce(fptr_t _fa, torch::Tensor &out) {
  auto fa = reinterpret_cast<mysync::Sync *>(_fa);
  switch (out.scalar_type()) {
    case at::ScalarType::Half: {
      fa->allreduce<half>(out.numel(), reinterpret_cast<half *>(out.data_ptr()));
      break;
    }
    default:
      throw std::runtime_error(
          "allreduce only supports float16");
  }
}