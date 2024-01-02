from pathlib import Path
import os
import time
import subprocess
from textwrap import dedent

import modal

r = lambda *args, **kwargs: subprocess.run(*args, shell=True, **kwargs)

image = (
    # Use a version of CUDA that's compatible w/ torch
    modal.Image.from_registry(
        "nvidia/cuda:12.2.2-devel-ubuntu22.04", add_python="3.11")
    .pip_install("sh", "torch==2.1.2", "ninja")
    #  build-essential is probably not needed
    .apt_install("git", "build-essential", "clang")
    # openmpi (nccl is already installed, so skip "libnccl-dev", "libnccl2")
    .apt_install("openmpi-bin", "openmpi-common", "libopenmpi-dev")
)

openmpi_src_image = (
    # Use a version of CUDA that's compatible w/ torch
    modal.Image.from_registry(
        "nvidia/cuda:12.2.2-devel-ubuntu22.04", add_python="3.11")
    .pip_install("sh", "torch==2.1.2", "ninja")
    .apt_install("git", "build-essential", "clang", "autotools-dev", "autoconf", "libtool")
    .run_commands(
        "git clone https://github.com/open-mpi/ompi.git",
        "cd ompi && git checkout v2.x && ./autogen.pl && ./configure",
        "cd ompi && make all && sudo make install"
    )
)

stub = modal.Stub()

# T4 (turing) has ~ instant results for 2,4 GPU count
t4_2 = modal.gpu.T4(count=2)
# A10G (ampere) has ~ instant for 2 GPU, ~ 1 minute for 4 GPU
# a10g = modal.gpu.A10G(count=4)
a10g = modal.gpu.A10G(count=2)

dirname = os.path.dirname(__file__)
csrc_dir = Path(dirname) / "csrc"

@stub.function(gpu=a10g, image=openmpi_src_image, cpu=8, 
               mounts=[modal.Mount.from_local_dir(csrc_dir, remote_path="/root/csrc")])
def build_pure_cuda_kernel():
    t0 = time.time()
    r("cd csrc/reference_allreduce && ./compile_combined.bash")
    print(f"Build time: {time.time() - t0:.2f}s")
    with open("hostfile.txt", "w") as f:
        f.write("localhost slots=2 max_slots=2")
    # print the mpi version
    r("mpirun --hostfile hostfile.txt --mca btl ^vader --allow-run-as-root -np 2 csrc/reference_allreduce/fastallreduce_test.bin")
    # r("mpirun --hostfile hostfile.txt  --allow-run-as-root -np 2 csrc/reference_allreduce/fastallreduce_test.bin")
    # r("mpirun --hostfile hostfile.txt --mca btl ^vader --allow-run-as-root -np 2 csrc/reference_allreduce/fastallreduce_test.bin")
    # r("mpirun --mca btl ^vader --allow-run-as-root -np 2 csrc/reference_allreduce/fastallreduce_test.bin")
    print(f"Total time: {time.time() - t0:.2f}s")


@stub.function(gpu="any", image=image)
def run_torch():
    print("Adding 1 + 1 on GPU")
    import torch
    x = torch.tensor([1.0]).cuda()
    r = x + x
    print(f"Finished: {r}")


@stub.function(gpu=t4_2, image=image, cpu=8)
def build_kernel_with_torch_bindings():
    t0 = time.time()
    r("git clone --depth 1 https://github.com/vedantroy/gpu_kernels.git")
    print(f"Clone time: {time.time() - t0:.2f}s")  # ~ 0.5s
    r("cd gpu_kernels/allreduce && python3 setup.py install")
    # modal (8 cpu) = ~70s
    # laptop = ~56s
    # vast (ryzen 9) = ~45s
    # vast (ryzen 9, no optimization) = ~40s
    print(f"Build time: {time.time() - t0:.2f}s")

    code = dedent("""
    import torch
    import cuda_experiments

    x = torch.ones(2, device="cuda")
    x_plus_x = cuda_experiments.add_one(x)
    torch.testing.assert_close(x_plus_x, x + x)
    """)

    r(f"echo '{code}' > gpu_kernels/allreduce/test.py")
    r("cd gpu_kernels/allreduce && python3 test.py")  # ~ 5s

    print(f"All time: {time.time() - t0:.2f}s")

# Topology, SMI, etc.

#     result = subprocess.run(["nvidia-smi", "topo", "-m"], stdout=subprocess.PIPE)
# subprocess.run(
#     [
#         "nvidia-smi",
#         "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used",
#         "--format=csv",
#     ]
# )
