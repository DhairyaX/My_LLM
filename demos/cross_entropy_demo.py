import torch
import torch.nn as nn

# Model's raw predictions (logits)
logits = torch.tensor([
    [2.5, 5.1, 0.4, 3.8]
])

# Correct answer index
target = torch.tensor([2])

criterion = nn.CrossEntropyLoss()

loss = criterion(logits, target)

print("Loss:")
print(loss)