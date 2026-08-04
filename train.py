import torch
from torch.utils.data import DataLoader

from config import Config
from data.dataset import ShakespeareDataset
from models.gpt import GPT

# -----------------------------
# Load Dataset
# -----------------------------

with open(
    "data/tiny_shakespeare.txt",
    "r",
    encoding="utf-8"
) as f:
    text = f.read()

dataset = ShakespeareDataset(
    text,
    Config.block_size
)

loader = DataLoader(
    dataset,
    batch_size=Config.batch_size,
    shuffle=True
)

# -----------------------------
# Build GPT
# -----------------------------

model = GPT(
    vocab_size=dataset.vocab_size,
    embedding_dim=Config.embedding_dim,
    num_heads=Config.num_heads,
    num_layers=Config.num_layers,
    forward_expansion=Config.forward_expansion,
    max_length=Config.block_size
)

print(model)

print("\nModel Created Successfully!\n")

# -----------------------------
# Test Forward Pass
# -----------------------------

x, y = next(iter(loader))

print("Input Shape :", x.shape)

logits = model(x)

print("Output Shape:", logits.shape)