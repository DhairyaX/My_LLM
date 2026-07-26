import torch
import torch.nn as nn

# Vocabulary
vocab = [
    "i",
    "love",
    "artificial",
    "intelligence",
    "machine",
    "learning"
]

vocab_size = len(vocab)

# Size of each embedding vector
embedding_dim = 8

# Create embedding layer
embedding = nn.Embedding(vocab_size, embedding_dim)

print("Embedding Layer:")
print(embedding)

print()

# Token IDs
token_ids = torch.tensor([0, 1, 2])

print("Token IDs:")
print(token_ids)

print()

# Convert IDs into embeddings
embedded = embedding(token_ids)

print("Embeddings:")
print(embedded)

print()

print("Shape:")
print(embedded.shape)