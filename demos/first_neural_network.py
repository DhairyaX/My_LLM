import torch
import torch.nn as nn

# Input features
x = torch.tensor([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0]
])

print("Input:")
print(x)
print()

# Create a linear layer
linear = nn.Linear(2, 3)

print("Linear Layer:")
print(linear)
print()

# Pass input through the layer
output = linear(x)

print("Output:")
print(output)
print()

print("Output Shape:")
print(output.shape)