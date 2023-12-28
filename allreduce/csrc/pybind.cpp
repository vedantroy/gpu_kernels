#pragma once
#include <torch/extension.h>
#include "add_one/add_one.h"
#include "reference_allreduce/fast_allreduce.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_one", &add_one, "Add one to all elements of the tensor");
}
