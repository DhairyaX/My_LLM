import torch

from config import Config
from data.dataset import ShakespeareDataset
from models.gpt import GPT

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# -----------------------
# Load Dataset
# -----------------------

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

# -----------------------
# Build Model
# -----------------------

model = GPT(
    vocab_size=full_dataset.vocab_size,
    embedding_dim=Config.embedding_dim,
    num_heads=Config.num_heads,
    num_layers=Config.num_layers,
    forward_expansion=Config.forward_expansion,
    max_length=Config.block_size
)

model.load_state_dict(
    torch.load(
    "checkpoints/gpt_model.pth",
    map_location=device
)
)

model.to(device)

# -----------------------
# Prompt
# -----------------------

prompt = "To be"

encoded = [
    full_dataset.stoi[c]
    for c in prompt
]

x = torch.tensor(
    encoded,
    dtype=torch.long
).unsqueeze(0).to(device)

generated = model.generate(
    x,
    max_new_tokens=100
)

output = "".join(
    full_dataset.itos[token.item()]
    for token in generated[0]
)

print()

print(output)