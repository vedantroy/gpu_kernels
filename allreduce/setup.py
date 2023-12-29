from typing import List

import os
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# T4 GPU
# os.environ['TORCH_CUDA_ARCH_LIST'] = '7.5'
# A2000
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'

ROOT_DIR = os.path.dirname(__file__)

print(f"# CPUs: {os.cpu_count()}")

extra_compile_args = {
    # "cxx": ["-g", "-O2", "-std=c++17"],
    # "nvcc": ["-O2", "-std=c++17", f"--threads={os.cpu_count()}"],
    "cxx": ["-g", "-std=c++17"],
    "nvcc": ["-std=c++17", f"--threads={os.cpu_count()}"],
}


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def get_requirements() -> List[str]:
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    return requirements


setup(
    name="cuda_experiments",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="cuda_experiments",
            sources=[
                "csrc/pybind.cpp",
                "csrc/add_one/add_one.cu",
                # "csrc/reference_allreduce/fast_allreduce.cu",
            ],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=get_requirements(),
)

# 38 secs nvcc
# pybind.cpp: 20s
# 

# delete object files
# find ./build/temp.linux-x86_64-cpython-311/csrc -name "*.o" | xargs rm

# ninja file:
# ninja_required_version = 1.3
# cxx = c++
# nvcc = /usr/local/cuda/bin/nvcc
# 
# cflags = -pthread -B /root/micromamba/envs/allreduce/compiler_compat -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /root/micromamba/envs/allreduce/include -fPIC -O2 -isystem /root/micromamba/envs/allreduce/include -fPIC -I/root/micromamba/envs/allreduce/lib/python3.11/site-packages/torch/include -I/root/micromamba/envs/allreduce/lib/python3.11/site-packages/torch/include/torch/csrc/api/include -I/root/micromamba/envs/allreduce/lib/python3.11/site-packages/torch/include/TH -I/root/micromamba/envs/allreduce/lib/python3.11/site-packages/torch/include/THC -I/usr/local/cuda/include -I/root/micromamba/envs/allreduce/include/python3.11 -c
# post_cflags = -g -std=c++17 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=cuda_experiments -D_GLIBCXX_USE_CXX11_ABI=0
# cuda_cflags = -I/root/micromamba/envs/allreduce/lib/python3.11/site-packages/torch/include -I/root/micromamba/envs/allreduce/lib/python3.11/site-packages/torch/include/torch/csrc/api/include -I/root/micromamba/envs/allreduce/lib/python3.11/site-packages/torch/include/TH -I/root/micromamba/envs/allreduce/lib/python3.11/site-packages/torch/include/THC -I/usr/local/cuda/include -I/root/micromamba/envs/allreduce/include/python3.11 -c
# cuda_post_cflags = -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -std=c++17 --threads=24 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=cuda_experiments -D_GLIBCXX_USE_CXX11_ABI=0 -gencode=arch=compute_86,code=sm_86
# cuda_dlink_post_cflags = 
# ldflags = 
# 
# rule compile
#   command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags
#   depfile = $out.d
#   deps = gcc
# 
# rule cuda_compile
#   depfile = $out.d
#   deps = gcc
#   command = $nvcc  $cuda_cflags -c $in -o $out $cuda_post_cflags
# 
# 
# 
# 
# 
# build /root/gpu_kernels/allreduce/build/temp.linux-x86_64-cpython-311/csrc/add_one/add_one.o: cuda_compile /root/gpu_kernels/allreduce/csrc/add_one/add_one.cu
# build /root/gpu_kernels/allreduce/build/temp.linux-x86_64-cpython-311/csrc/pybind.o: compile /root/gpu_kernels/allreduce/csrc/pybind.cpp