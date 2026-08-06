import torch
import torch.nn as nn

torch.manual_seed(42)

embedding_dim = 8

# Example embeddings for 4 tokens
embeddings = torch.randn(4, embedding_dim)

print("Embeddings Shape:")
print(embeddings.shape)

# Three different linear layers
query_layer = nn.Linear(embedding_dim, embedding_dim)

key_layer = nn.Linear(embedding_dim, embedding_dim)

value_layer = nn.Linear(embedding_dim, embedding_dim)

Q = query_layer(embeddings)
K = key_layer(embeddings)
V = value_layer(embeddings)

print("\nQuery Shape:", Q.shape)
print("Key Shape:", K.shape)
print("Value Shape:", V.shape)

print("\nFirst Query Vector:")
print(Q[0])

print("\nFirst Key Vector:")
print(K[0])

print("\nFirst Value Vector:")
print(V[0])