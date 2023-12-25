from typing import List

import os
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = os.path.dirname(__file__)

extra_compile_args = {
    "cxx": ["-g", "-O2", "-std=c++17"],
    "nvcc": ["-O2", "-std=c++17"],
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
                "csrc/add_one.cu", 
            ],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=get_requirements(),
)