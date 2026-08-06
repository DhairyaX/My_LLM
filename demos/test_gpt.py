import torch

from gpt import GPT

torch.manual_seed(42)

vocab_size = 100
embedding_dim = 32
num_heads = 4
num_layers = 2
forward_expansion = 4
max_length = 20

model = GPT(
    vocab_size,
    embedding_dim,
    num_heads,
    num_layers,
    forward_expansion,
    max_length
)

x = torch.randint(
    0,
    vocab_size,
    (2,10)
)

output = model(x)

print("Input Shape :", x.shape)
print("Output Shape:", output.shape)