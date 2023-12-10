# A reference implementation of the dequantization in pure Pytorch
import torch

# Generate random data
M = torch.randint(0, 1000, (1,)).item()
N = K = 4096
pack_num = 8
group_size = 128
print(f"Testing with M={M}")
int32_bounds = (torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max)
inputs = torch.randn((M, K), dtype=torch.float16, device="cuda")
qweight = torch.randint(*int32_bounds, (N, K // pack_num), dtype=torch.int32, device="cuda")
scales = 0.001 * torch.abs(torch.randn((N, K // group_size), dtype=torch.float16, device="cuda"))
qzeros = torch.randint(*int32_bounds, (N, K // group_size // pack_num), dtype=torch.int32, device="cuda")

def matmul_simple(
    a, qw, qzeros, scales
):
    N = 4096
    pack_num = 8
    group_size = 128

    M, K = a.shape
    # ASSUMPTION:
    # all quantization / packing is done along the channel dimension
    # (channels become lower-resolution, but the # of channels is the same)
    assert qw.shape == (N, K // pack_num)
    assert qzeros.shape == (N, K // group_size // pack_num)
    assert scales.shape == (N, K // group_size)

    print("dequantizing matrix ...")

    # dequant small tile
    n_rows_to_dequant = 64
    K2 = 32

    # dequant full matrix
    # n_rows_to_dequant = N
    # K2 = K
    dequant_matrix = torch.zeros((n_rows_to_dequant, K2), dtype=torch.float32, device=a.device)

    from tqdm import tqdm

    for row in tqdm(range(n_rows_to_dequant)):
        dequant_row = torch.zeros((K2, ), dtype=torch.float32, device=a.device)
        for col in range(K2):
            group_idx = col // group_size
            scale = scales[row][group_idx].to(torch.float32)
            qzero = qzeros[row][group_idx // pack_num]
            qweight = qw[row][col // pack_num] 

            # assert col // group_size == 0
            # assert scale == scales[row][0]
            # assert qzero == qzeros[row][0]
            # assert qweight in [qw[row][0], qw[row][1], qw[row][2], qw[row][3]]

            qzero_unpacked = ((qzero >> (4 * (group_idx % pack_num))) & 0xF).to(torch.float32)
            qweight_unpacked = ((qweight >> (4 * (col % pack_num))) & 0xF).to(torch.float32)
            dequant = scale * (qweight_unpacked - qzero_unpacked)
            dequant_row[col] = dequant
        dequant_matrix[row] = dequant_row
    torch.cuda.synchronize()
    print("finished dequantizing ...")
    return dequant_matrix
 
manual_dequant = matmul_simple(inputs, qweight, qzeros, scales)