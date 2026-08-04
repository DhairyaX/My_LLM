import torch

from models.multi_head_attention import MultiHeadAttention
torch.manual_seed(42)

batch_size = 2
sequence_length = 5
embedding_dim = 16
num_heads = 4

x = torch.randn(batch_size, sequence_length, embedding_dim)

mha = MultiHeadAttention(
    embedding_dim=embedding_dim,
    num_heads=num_heads
)

output = mha(x)

print("Input Shape :", x.shape)
print("Output Shape:", output.shape)