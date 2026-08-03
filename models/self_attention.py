import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):

    def __init__(self, embedding_dim):

        super().__init__()

        self.embedding_dim = embedding_dim

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, embeddings):

        # Step 1
        Q = self.query(embeddings)

        # Step 2
        K = self.key(embeddings)

        # Step 3
        V = self.value(embeddings)

        # Step 4
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # Step 5
        scores = scores / math.sqrt(self.embedding_dim)

        # Step 6
        attention_weights = F.softmax(scores, dim=-1)

        # Step 7
        output = torch.matmul(attention_weights, V)

        return output, attention_weights