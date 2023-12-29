import time
import modal
import subprocess

image = (
    # Use a version of CUDA that's compatible w/ torch
    modal.Image.from_registry("nvidia/cuda:12.2.2-devel-ubuntu22.04", add_python="3.11")
    .pip_install("sh", "torch==2.1.2", "ninja")
    #  build-essential is probably not needed
    .apt_install("git", "build-essential", "clang")
)
stub = modal.Stub()

@stub.function(gpu="any", image=image)
def run_torch():
    print("Adding 1 + 1 on GPU")
    import torch
    x = torch.tensor([1.0]).cuda()
    r = x + x
    print(f"Finished: {r}")

# T4 (turing) has ~ instant results for 2,4 GPU count
t4_2 = modal.gpu.T4(count=2)
# A10G (ampere) has ~ instant for 2 GPU, ~ 1 minute for 4 GPU
a10g = modal.gpu.A10G(count=4)

r = lambda *args, **kwargs: subprocess.run(*args, shell=True, **kwargs)

@stub.function(gpu=t4_2, image=image, cpu=8)
def build_kernel():
    t0 = time.time()
    r("git clone --depth 1 https://github.com/vedantroy/gpu_kernels.git")
    print(f"Clone time: {time.time() - t0:.2f}s")
    r("cd gpu_kernels/allreduce && python3 setup.py install")
    print(f"Build time: {time.time() - t0:.2f}s")