#! /usr/bin/env bash

# Run `mpicc -show` to find include directories

# https://docs.ccv.brown.edu/oscar/gpu-computing/mpi-cuda
# https://anhnguyen.me/2013/12/how-to-mix-mpi-and-cuda-in-a-single-program/
# nvcc -I/usr/mpi/gcc/openmpi-1.4.6/include -L/usr/mpi/gcc/openmpi-1.4.6/lib64 -lmpi spaghetti.cu -o program
script_dir=$(dirname "$0")
# -I/usr/include => path to nccl include directory
# -L/usr/lib/x86_64-linux-gnu => path to ncl libraries
nvcc -I/usr/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -L/usr/lib/x86_64-linux-gnu -lnccl -gencode=arch=compute_86,code=sm_86 "$script_dir/fast_allreduce_test.cu" -o fastallreduce_test.bin
