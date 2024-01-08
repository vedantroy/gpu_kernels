#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>
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

int main(int argc, char **argv) {
  #define N_ELEMENTS (1024 * 8)
  #define DTYPE half

    DTYPE *input_buf;
    // allocate N elements
    CUDACHECK(cudaMalloc(&input_buf, N_ELEMENTS * sizeof(DTYPE)));

    // This is wrong b/c we are setting each byte to 1
    // but fp16 values are 2 bytes
    // CUDACHECK(cudaMemset(input_buf, 1, N_ELEMENTS));

    DTYPE *input_buf_cpu = new DTYPE[N_ELEMENTS];
    for (int i = 0; i < N_ELEMENTS; i++) {
        input_buf_cpu[i] = __float2half(1.0f);
    }

    // Copy from CPU to GPU
    CUDACHECK(cudaMemcpy(input_buf, input_buf_cpu, N_ELEMENTS * sizeof(DTYPE), cudaMemcpyHostToDevice));

    // mem copy to cpu and print first element
    DTYPE *input_buf_cpu2 = new DTYPE[N_ELEMENTS];
    CUDACHECK(cudaMemcpy(input_buf_cpu, input_buf, N_ELEMENTS * sizeof(DTYPE), cudaMemcpyDeviceToHost));
    printf("input_buf[0] = %f\n", __half2float(input_buf_cpu2[0]));

  return EXIT_SUCCESS;
}
