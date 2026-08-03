import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# Tiny dataset (XOR-like)
X = torch.tensor([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
])

y = torch.tensor([0, 1, 1, 0])

# Model
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 2)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

epochs = 100

for epoch in range(epochs):

    # Forward
    outputs = model(X)

    # Loss
    loss = criterion(outputs, y)

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1} | Loss = {loss.item():.4f}")
        

print("\nPredictions")

with torch.no_grad():

    outputs = model(X)

    predictions = torch.argmax(outputs, dim=1)

    print("Predicted:", predictions)
    print("Actual   :", y)