#pragma once
// #include <torch/extension.h>
#include "add_one.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_one", &add_one, "Add one to all elements of the tensor");
}
