import torch
import torch.nn as nn
import torch.optim as optim

# For reproducibility
torch.manual_seed(42)

# A tiny neural network
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 2)
)

# Dummy input
x = torch.tensor([
    [1.0, 2.0]
])

# Correct answer
target = torch.tensor([1])

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.1)

# Forward Pass
output = model(x)

print("Prediction:")
print(output)

loss = criterion(output, target)

print("\nLoss:")
print(loss)

# Backpropagation
optimizer.zero_grad()

loss.backward()

print("\nGradient of First Layer:")
print(model[0].weight.grad)

# Update Weights
optimizer.step()

print("\nWeights Updated Successfully!")