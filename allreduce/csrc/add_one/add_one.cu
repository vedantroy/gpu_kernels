#include <torch/extension.h>
#include "add_one.h"

__global__ void add_one_kernel(const float* in_data, float* out_data, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        out_data[index] = in_data[index] + 1.0;
    }
}

torch::Tensor add_one(torch::Tensor input) {
    TORCH_CHECK(input.type().is_cuda() && input.type().scalarType() == at::ScalarType::Float, "input must be a CUDA float tensor");
    auto output = torch::empty_like(input);
    const auto size = input.numel();
    const int threads = 1024;
    const int blocks = (size + threads - 1) / threads;

    add_one_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}



