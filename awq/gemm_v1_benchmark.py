# Taken from the Triton matmul tutorial
import os
import torch
import triton

import awq_inference_engine as ie
from gemm_kernel_v1 import quant_matmul

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M'],  # Argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['cuda', 'triton'],
        # Label name for the lines
        line_names=["cuda", "Triton"],
        # Line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    )
)
def benchmark(M, provider):
    N = K = 4096
    pack_num = 8
    group_size = 128
    int32_bounds = (torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max)
    inputs = torch.randn((M, K), dtype=torch.float16, device="cuda")
    qweight = torch.randint(*int32_bounds, (N, K // pack_num), dtype=torch.int32, device="cuda")
    scales = 0.001 * torch.abs(torch.randn((N, K // group_size), dtype=torch.float16, device="cuda"))
    qzeros = torch.randint(*int32_bounds, (N, K // group_size // pack_num), dtype=torch.int32, device="cuda")

    if provider == 'triton':
        trans = lambda x: x.T.contiguous()
        qweight = trans(qweight)
        qzeros = trans(qzeros)
        scales = trans(scales)

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cuda':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: ie.gemm_forward_cuda(inputs, qweight, scales, qzeros, group_size, 8), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: quant_matmul(inputs, qweight, qzeros, scales, M=M, N=N, K=K, pack_num=pack_num, group_size=group_size), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

parent_dir = os.path.dirname(os.path.realpath(__file__))
benchmark.run(show_plots=True, print_data=True, save_path=parent_dir)