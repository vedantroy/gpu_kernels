#! /usr/bin/env bash
set -euxo pipefail

  # Get GPU name
  gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader)

  # Set architecture based on GPU name
  case $gpu_name in
    *T4*)
      arch=compute_75
      code=sm_75
      ;;
    *A10*)
      arch=compute_80
      code=sm_80
      ;;
    *V100*)
      arch=compute_70
      code=sm_70
      ;;
    *A2000*)
      arch=compute_86
      code=sm_86
      ;;
    *V100*)
      arch=compute_70
      code=sm_70
      ;;
    *)
      echo "Unsupported GPU: $gpu_name"
      exit 1
      ;;
  esac

echo "GPU: $gpu_name"
echo "Architecture: $arch"
echo "Compute capability: $code"

# Directory of the script
script_dir=$(dirname "$0")

# nvcc command with dynamic architecture
nvcc -I/usr/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -L/usr/lib/x86_64-linux-gnu -lnccl -gencode=arch=$arch,code=$code "$script_dir/fast_allreduce_test.cu" -o fastallreduce_test.bin


# # Run `mpicc -show` to find include directories
# 
# # https://docs.ccv.brown.edu/oscar/gpu-computing/mpi-cuda
# # https://anhnguyen.me/2013/12/how-to-mix-mpi-and-cuda-in-a-single-program/
# # nvcc -I/usr/mpi/gcc/openmpi-1.4.6/include -L/usr/mpi/gcc/openmpi-1.4.6/lib64 -lmpi spaghetti.cu -o program
# script_dir=$(dirname "$0")
# # -I/usr/include => path to nccl include directory
# # -L/usr/lib/x86_64-linux-gnu => path to ncl libraries
# nvcc -I/usr/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -L/usr/lib/x86_64-linux-gnu -lnccl -gencode=arch=compute_86,code=sm_86 "$script_dir/fast_allreduce_test.cu" -o fastallreduce_test.bin
