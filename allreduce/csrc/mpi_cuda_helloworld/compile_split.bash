
nvcc -c csrc/mpi_cuda_helloworld/split_kernel.cu -o csrc/mpi_cuda_helloworld/split_kernel.o -gencode=arch=compute_86,code=sm_86
mpicc -c ./csrc/mpi_cuda_helloworld/split_driver.c -o ./csrc/mpi_cuda_helloworld/split_driver.o