import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

embedding_dim = 8
num_tokens = 4

# Fake embeddings
embeddings = torch.randn(num_tokens, embedding_dim)

# Q, K, V
query = nn.Linear(embedding_dim, embedding_dim)
key = nn.Linear(embedding_dim, embedding_dim)
value = nn.Linear(embedding_dim, embedding_dim)

Q = query(embeddings)
K = key(embeddings)
V = value(embeddings)

# Step 1: Attention Scores
scores = torch.matmul(Q, K.T)

print("Raw Scores:\n")
print(scores)

# Step 2: Scale
scaled_scores = scores / math.sqrt(embedding_dim)

print("\nScaled Scores:\n")
print(scaled_scores)

# Step 3: Softmax
attention_weights = F.softmax(scaled_scores, dim=1)

print("\nAttention Weights:\n")
print(attention_weights)

print("\nRow Sums:")
print(attention_weights.sum(dim=1))


attention_output = torch.matmul(attention_weights, V)

print("\nAttention Output:\n")
print(attention_output)

print("\nOutput Shape:")
print(attention_output.shape)