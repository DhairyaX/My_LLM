import torch
import torch.nn as nn

from transformer_block import TransformerBlock


class GPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_heads,
        num_layers,
        forward_expansion,
        max_length
    ):

        super().__init__()

        self.word_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = nn.Embedding(
            max_length,
            embedding_dim
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim,
                    num_heads,
                    forward_expansion
                )
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, x):

        batch_size, seq_length = x.shape

        positions = torch.arange(
            0,
            seq_length
        ).expand(batch_size, seq_length)

        x = (
            self.word_embedding(x)
            +
            self.position_embedding(positions)
        )

        for layer in self.layers:
            x = layer(x)

        logits = self.fc_out(x)

        return logits