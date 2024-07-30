import numpy as np
import torch
import einops as rearrange

def test_reshape():
    np_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32)
    torch_tensor = torch.from_numpy(np_array)
    reshaped_tensor = rearrange.rearrange(torch_tensor, '(b h) w -> b w h',b = 2)
    # reshaped_tensor = torch_tensor.reshape(2, 2 ,3).permute(0, 2, 1)

    print(reshaped_tensor)

if __name__ == "__main__":
    test_reshape()