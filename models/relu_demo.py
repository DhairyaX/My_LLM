import torch
import torch.nn as nn

# Input tensor
x = torch.tensor([
    [-2.0, -1.0, 0.0],
    [1.0, 2.0, -3.0],
    [4.0, -5.0, 6.0]
])

print("Input:")
print(x)

relu = nn.ReLU()

output = relu(x)

print("\nOutput:")
print(output)
