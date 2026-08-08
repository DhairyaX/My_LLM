import torch
import torch.nn as nn
from config import Config

from models.transformer_block import TransformerBlock


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

        # -----------------------------
        # Token Embeddings
        # -----------------------------

        self.word_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        # -----------------------------
        # Position Embeddings
        # -----------------------------

        self.position_embedding = nn.Embedding(
            max_length,
            embedding_dim
        )

        # -----------------------------
        # Transformer Blocks
        # -----------------------------

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim,
                    num_heads,
                    forward_expansion,
                    Config.dropout
                )
                for _ in range(num_layers)
            ]
        )

        # -----------------------------
        # Output Layer
        # -----------------------------

        self.fc_out = nn.Linear(
            embedding_dim,
            vocab_size
        )

        # -----------------------------
        # Initialize Weights
        # -----------------------------

        self.apply(self._init_weights)


    # -----------------------------
    # Weight Initialization
    # -----------------------------

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):

            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:

                torch.nn.init.zeros_(
                    module.bias
                )

        elif isinstance(module, nn.Embedding):

            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )


    # -----------------------------
    # Forward Pass
    # -----------------------------

    def forward(self, x):

        batch_size, seq_length = x.shape

        positions = torch.arange(
            0,
            seq_length,
            device=x.device
        ).expand(
            batch_size,
            seq_length
        )

        x = (
            self.word_embedding(x)
            +
            self.position_embedding(positions)
        )

        for layer in self.layers:

            x = layer(x)

        logits = self.fc_out(x)

        return logits


    # -----------------------------
    # Text Generation
    # -----------------------------

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=0.8,
        top_k=10
    ):

        self.eval()

        for _ in range(max_new_tokens):

            # Keep only the available context window
            idx_cond = idx[
                :,
                -self.position_embedding.num_embeddings:
            ]

            # Forward pass
            logits = self(idx_cond)

            # Get prediction for the final token
            logits = logits[:, -1, :]

            # -----------------------------
            # Temperature
            # -----------------------------

            logits = logits / temperature

            # -----------------------------
            # Top-k Sampling
            # -----------------------------

            if top_k is not None:

                values, _ = torch.topk(
                    logits,
                    top_k
                )

                min_value = values[:, -1].unsqueeze(-1)

                logits = torch.where(
                    logits < min_value,
                    torch.full_like(
                        logits,
                        float("-inf")
                    ),
                    logits
                )

            # -----------------------------
            # Convert to Probabilities
            # -----------------------------

            probs = torch.softmax(
                logits,
                dim=-1
            )

            # -----------------------------
            # Sample Next Token
            # -----------------------------

            next_token = torch.multinomial(
                probs,
                num_samples=1
            )

            # -----------------------------
            # Add Token to Sequence
            # -----------------------------

            idx = torch.cat(
                (idx, next_token),
                dim=1
            )

        return idx