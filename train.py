import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import Config
from data.dataset import ShakespeareDataset
from models.gpt import GPT

# -----------------------------
# Device
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)

# -----------------------------
# Load Dataset
# -----------------------------

with open(
    "data/tiny_shakespeare.txt",
    "r",
    encoding="utf-8"
) as f:
    text = f.read()

from torch.utils.data import Subset

full_dataset = ShakespeareDataset(
    text,
    Config.block_size
)

dataset = Subset(full_dataset, range(5000))

loader = DataLoader(
    dataset,
    batch_size=Config.batch_size,
    shuffle=True
)

print("Total batches:", len(loader))

# -----------------------------
# Build Model
# -----------------------------

model = GPT(
    vocab_size=full_dataset.vocab_size,
    embedding_dim=Config.embedding_dim,
    num_heads=Config.num_heads,
    num_layers=Config.num_layers,
    forward_expansion=Config.forward_expansion,
    max_length=Config.block_size
)

model = model.to(device)

# -----------------------------
# Loss & Optimizer
# -----------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=Config.learning_rate
)

# -----------------------------
# One Training Step
# -----------------------------
# -----------------------------
# Training Loop
# -----------------------------

model.train()

for epoch in range(Config.epochs):

    total_loss = 0

    for batch_idx, (x, y) in enumerate(loader):

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(

            logits.view(-1, full_dataset.vocab_size),

            y.view(-1)

        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:

            print(
                f"Epoch {epoch+1} | "
                f"Batch {batch_idx+1} | "
                f"Loss {loss.item():.4f}"
            )

    avg_loss = total_loss / len(loader)

    print()

    print(
        f"Epoch {epoch+1} Complete "
        f"| Average Loss = {avg_loss:.4f}"
    )

    print("-" * 50)


# -----------------------------
# Save Model
# -----------------------------

import os

os.makedirs("checkpoints", exist_ok=True)

torch.save(
    model.state_dict(),
    "checkpoints/gpt_model.pth"
)

print("\nModel saved successfully!")
