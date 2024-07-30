import pytest
import torch

def test_veiw():
    tensor = torch.randn(6,32,8,8)
    tensor = tensor.view(-1,2048)
    print(tensor.shape)
    