import torch

from config import Config
from data.dataset import ShakespeareDataset
from models.gpt import GPT


# -----------------------------
# Device
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -----------------------------
# Load Dataset
# -----------------------------

with open(
    "data/tiny_shakespeare.txt",
    "r",
    encoding="utf-8"
) as f:
    text = f.read()

full_dataset = ShakespeareDataset(
    text,
    Config.block_size
)


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


# -----------------------------
# Load Trained Model
# -----------------------------

model.load_state_dict(
    torch.load(
        "checkpoints/best_model.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Model loaded successfully!")


# -----------------------------
# Prompt
# -----------------------------

prompt = input("Enter your prompt: ")

encoded = [
    full_dataset.stoi[c]
    for c in prompt
]

x = torch.tensor(
    encoded,
    dtype=torch.long
).unsqueeze(0).to(device)


# -----------------------------
# Generate Text
# -----------------------------

generated = model.generate(
    x,
    max_new_tokens=100,
    temperature=0.8,
    top_k=10
)


# -----------------------------
# Decode Tokens
# -----------------------------

output = "".join(
    full_dataset.itos[token.item()]
    for token in generated[0]
)


# -----------------------------
# Print Result
# -----------------------------

print("\nGenerated Text:")
print("-----------------------------")
print(output)