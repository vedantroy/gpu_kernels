#include <torch/extension.h>

#include "add_one/add_one.h"
#include "my_allreduce/allreduce_torch_bindings.cu"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add_one", &add_one, "Add one to all elements of the tensor");

  m.def("init_ar", &init_ar, "init_ar");
  m.def("allreduce", &allreduce, "allreduce");
  m.def("register_buffer", &register_buffer, "register_buffer");
}
