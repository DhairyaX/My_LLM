import torch
import torch.nn as nn

torch.manual_seed(42)

# Batch of 4 samples
x = torch.randn(4, 5)

print("Input Shape:", x.shape)

model = nn.Sequential(
    nn.Linear(5, 8),
    nn.ReLU(),
    nn.Linear(8, 4)
)

output = model(x)

print("Output Shape:", output.shape)

print("\nOutput:")
print(output)

