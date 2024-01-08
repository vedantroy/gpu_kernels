#! /usr/bin/env bash
set -euxo pipefail

if command -v nvidia-smi &> /dev/null
then
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
    *4090*)
      arch=compute_90
      code=sm_90
      ;;
    *A4000*)
      arch=compute_86
      code=sm_86
      ;;
    *)
      echo "Unsupported GPU: $gpu_name"
      exit 1
      ;;
  esac
else
  # Default to A2000 if nvidia-smi does not exist
  gpu_name="A2000"
  arch=compute_86
  code=sm_86
fi

echo "GPU: $gpu_name"
echo "Architecture: $arch"
echo "Compute capability: $code"

# Directory of the script
script_dir=$(dirname "$0")

# nvcc command with dynamic architecture
nvcc -I/usr/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -L/usr/lib/x86_64-linux-gnu -lnccl -gencode=arch=$arch,code=$code "$script_dir/allreduce_test.cu" -o allreduce_test.bin --expt-relaxed-constexpr -G
