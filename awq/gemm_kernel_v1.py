# Please give credit to "Vedant Roy"
import torch
import triton
import triton.language as tl


@triton.jit
def quant_matmul_kernel(
    # Pointers to matrices
    a_ptr, qw_ptr, c_ptr, scales_ptr, zeros_ptr,
    # Matrix dimensions
    M, N, K, 
    # Quantization parameters
    group_size,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, 
):
    """
    Kernel for computing the matmul C = A x qw

    a: (M, K)
    qw: (K // pack_num, N)
    scales: (K // group_size, N)
    qzeros: (K // group_size // pack_num, N)
    """

    stride_zeros_k = N
    stride_scales_k = N
    stride_a_m = K
    stride_qw_k = N

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)  # (K,)
    qw_shifter = (offs_k % 8) * 4

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_offs = (k * BLOCK_SIZE_K) + (offs_am[:, None] * stride_a_m + offs_k[None, :])  # (M, K)
        a = tl.load( a_ptr + a_offs)

        qw_offs = (((k * BLOCK_SIZE_K) + offs_k[:, None]) // 8) * stride_qw_k + offs_bn[
            None, :
        ]  # (K, N)
        qw_packed = tl.load(qw_ptr + qw_offs)  # (K, N)

        qw_unpacked = (qw_packed >> qw_shifter[:, None]) & 0xF

        k_iters_per_quant_group = group_size // BLOCK_SIZE_K
        grp_idx = k // k_iters_per_quant_group

        col_offs = offs_bn
        scales = tl.load(scales_ptr + (stride_scales_k * grp_idx) + col_offs)  # (N,)

        packed_zeros = tl.load(
            zeros_ptr + stride_zeros_k * (grp_idx // 8) + col_offs
        )  # (N,)
        unpacked_zeros = (packed_zeros >> ((grp_idx % 8) * 4)) & 0xF

        dequantized = scales[None, :].to(tl.float32) * (
            qw_unpacked.to(tl.float32) - unpacked_zeros[None, :].to(tl.float32)
        )
        accumulator += tl.dot(a, dequantized.to(tl.float16))
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    stride_cm = N
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def quant_matmul(a, qw, qzeros, scales, *, M, N, K, pack_num, group_size):
    c = torch.empty((M, N), dtype=torch.float16, device=a.device)

    assert qw.shape == (K // pack_num, N)
    assert qzeros.shape == (K // group_size // pack_num, N)
    assert scales.shape == (K // group_size, N)
    assert all(x.is_contiguous() for x in [a, qw, c, qzeros, scales])
    # BLOCK_SIZE_K has possible values of 32, 64
    # group_size, K must be divisible by BLOCK_SIZE_K
    assert group_size % 64 == 0, f"group_size {group_size} is not a multiple of 64"
    assert K % 64 == 0, f"K {K} is not a multiple of 64"

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    grid_1d = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    quant_matmul_kernel[grid_1d](
        a_ptr=a,
        qw_ptr=qw,
        c_ptr=c,
        scales_ptr=scales,
        zeros_ptr=qzeros,
        M=M,
        N=N,
        K=K,
        group_size=group_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=1,
    )
    return c

if __name__ == "__main__":
    # Tested with AWQ commit f0b4b68004f76d562658143cddea5aad8c1b8266
    import awq_inference_engine as ie

    M = torch.randint(0, 1000, (1,)).item()
    print(f"Testing with M={M}")

    # M = 128
    N = K = 4096
    pack_num = 8
    group_size = 128

    int32_bounds = (torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max)
    inputs = torch.randn((M, K), dtype=torch.float16, device="cuda")
    qweight = torch.randint(*int32_bounds, (N, K // pack_num), dtype=torch.int32, device="cuda")
    # TODO: Check what the proper magnitude of scales is
    # scales is always positive & needs to be very small (otherwise this test fails) ??
    # Sample stats of scales from a single real layer
    # min: 0.0004451274871826172
    # max: 0.00801849365234375
    # std: 0.0006480216979980469
    # mean: 0.0015420913696289062)
    scales = 0.001 * torch.abs(torch.randn((N, K // group_size), dtype=torch.float16, device="cuda"))
    qzeros = torch.randint(*int32_bounds, (N, K // group_size // pack_num), dtype=torch.int32, device="cuda")

    out_cuda = ie.gemm_forward_cuda(inputs, qweight, scales, qzeros, group_size, 8)
    trans = lambda x: x.T.contiguous()
    out_triton = quant_matmul(inputs, trans(qweight), trans(qzeros), trans(scales), M=M, N=N, K=K, pack_num=pack_num, group_size=group_size)

    torch.testing.assert_close(out_cuda, out_triton, rtol=1e-3, atol=1e-3)