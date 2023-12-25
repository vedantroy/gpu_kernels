import torch
import cuda_experiments

tensor = torch.randn(5, device='cuda')
result = cuda_experiments.add_one(tensor)
torch.testing.assert_close(result, tensor + 1)
