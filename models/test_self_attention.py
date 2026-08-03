import torch

from self_attention import SelfAttention

torch.manual_seed(42)

embedding_dim = 8
num_tokens = 5

embeddings = torch.randn(num_tokens, embedding_dim)

attention = SelfAttention(embedding_dim)

# output = attention(embeddings)

# print("Input Shape :", embeddings.shape)
# print("Output Shape:", output.shape)

# print("\nOutput:")
# print(output)

output, weights = attention(embeddings)

print("\nAttention Weights:")
print(weights)

print("\nShape:")
print(weights.shape)