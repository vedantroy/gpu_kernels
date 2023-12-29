import subprocess
import time
import sh
import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.3.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install("sh", "ninja", "torch==2.1.2")
)
stub = modal.Stub()

# T4 (turing) has ~ instant results for 2,4 GPU count
t4 = modal.gpu.T4(count=2)
# A10G (ampere) has ~ instant for 2 GPU, ~ 1 minute for 4 GPU
a10g = modal.gpu.A10G(count=4)


@stub.function(gpu=t4, image=image, cpu=8)
def gpu_stats():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used",
            "--format=csv",
        ],
        stdout=subprocess.PIPE,
    )
    print(result.stdout.decode("utf-8"))

    result = subprocess.run(["nvidia-smi", "topo", "-m"], stdout=subprocess.PIPE)
    print(result.stdout.decode("utf-8"))


@stub.function(gpu=t4, image=image, cpu=8)
def build_on_device():
    t0 = time.time()

    subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used",
            "--format=csv",
        ]
    )

    REPO_URL = "https://github.com/vedantroy/gpu_kernels.git"
    sh.git.clone("--depth", "1", REPO_URL)
    sh.python3("-c", "import torch; x = torch.tensor([1.0]).cuda(); print(x + x)")
    sh.python3("setup.py", "install", _cwd="gpu_kernels/allreduce")
    print(f"Build time: {time.time() - t0:.2f}s")
