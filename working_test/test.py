import torch

output_size = 3

batch_size = 2

anchor_per_scale = 3

y = torch.arange(output_size, dtype=torch.int32).unsqueeze(1).repeat(1, output_size)
x = torch.arange(output_size, dtype=torch.int32).unsqueeze(0).repeat(output_size, 1)

xy_grid = torch.stack([x, y], dim=-1)
print(xy_grid)

xy_grid = xy_grid.unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, anchor_per_scale, 1).float()
print(xy_grid)

a = 1