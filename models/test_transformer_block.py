import torch

from transformer_block import TransformerBlock

torch.manual_seed(42)

batch_size = 2
sequence_length = 5
embedding_dim = 16
num_heads = 4
forward_expansion = 4

x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim
)

block = TransformerBlock(
    embedding_dim,
    num_heads,
    forward_expansion
)

output = block(x)

print("Input Shape :", x.shape)
print("Output Shape:", output.shape)

print("\nOutput:")
print(output)