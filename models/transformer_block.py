import torch
import torch.nn as nn

from models.multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        forward_expansion,
        dropout=0.1
    ):

        super().__init__()

        # -----------------------------
        # Multi-Head Attention
        # -----------------------------

        self.attention = MultiHeadAttention(
            embedding_dim,
            num_heads
        )

        # -----------------------------
        # Layer Normalization
        # -----------------------------

        self.norm1 = nn.LayerNorm(
            embedding_dim
        )

        self.norm2 = nn.LayerNorm(
            embedding_dim
        )

        # -----------------------------
        # Feed Forward Network
        # -----------------------------

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

        # -----------------------------
        # Dropout
        # -----------------------------

        self.dropout = nn.Dropout(
            dropout
        )


    # -----------------------------
    # Forward Pass
    # -----------------------------

    def forward(self, x):

        # Attention
        attention = self.attention(x)

        x = self.norm1(
            x + self.dropout(attention)
        )

        # Feed Forward
        forward = self.feed_forward(x)

        out = self.norm2(
            x + self.dropout(forward)
        )

        return out