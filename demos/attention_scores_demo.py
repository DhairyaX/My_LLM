import torch
import torch.nn as nn

torch.manual_seed(42)

embedding_dim = 8
num_tokens = 4

# Fake token embeddings
embeddings = torch.randn(num_tokens, embedding_dim)

# Q, K, V layers
query = nn.Linear(embedding_dim, embedding_dim)
key = nn.Linear(embedding_dim, embedding_dim)
value = nn.Linear(embedding_dim, embedding_dim)

Q = query(embeddings)
K = key(embeddings)
V = value(embeddings)

# Compute attention scores
scores = torch.matmul(Q, K.T)

print("Q Shape:", Q.shape)
print("K Shape:", K.shape)
print()

print("Attention Scores Shape:", scores.shape)
print(scores)

print("\nFirst Token Attention Scores:")
print(scores[0])