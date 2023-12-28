import modal

# Compile it here & copy into Docker container
# File mounts ...

# Build it there ...

image = modal.Image.debian_slim()
stub = modal.Stub()

# T4 (turing) has ~ instant results for 2,4 GPU count
t4 = modal.gpu.T4(count=2)
# A10G (ampere) has ~ instant for 2 GPU, ~ 1 minute for 4 GPU
a10g = modal.gpu.A10G(count=4)

@stub.function(gpu=t4, image=image)
def gpu_stats():
    import subprocess

    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used', '--format=csv'], stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))

    result = subprocess.run(['nvidia-smi', 'topo', '-m'], stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))