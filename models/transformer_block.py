import torch
import torch.nn as nn

from multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):

    def __init__(self, embedding_dim, num_heads, forward_expansion):

        super().__init__()

        self.attention = MultiHeadAttention(
            embedding_dim,
            num_heads
        )

        self.norm1 = nn.LayerNorm(embedding_dim)

        self.norm2 = nn.LayerNorm(embedding_dim)

        self.feed_forward = nn.Sequential(

            nn.Linear(
                embedding_dim,
                forward_expansion * embedding_dim
            ),

            nn.GELU(),

            nn.Linear(
                forward_expansion * embedding_dim,
                embedding_dim
            )

        )

    def forward(self, x):

        attention = self.attention(x)

        x = self.norm1(x + attention)

        forward = self.feed_forward(x)

        out = self.norm2(x + forward)

        return out