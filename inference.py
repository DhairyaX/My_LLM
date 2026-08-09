import torch

from config import Config
from data.dataset import ShakespeareDataset
from models.gpt import GPT


class MiniGPT:

    def __init__(
        self,
        model_path="checkpoints/best_model.pth"
    ):

        # -----------------------------
        # Device
        # -----------------------------

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # -----------------------------
        # Load Dataset / Vocabulary
        # -----------------------------

        with open(
            "data/tiny_shakespeare.txt",
            "r",
            encoding="utf-8"
        ) as f:
            text = f.read()

        self.dataset = ShakespeareDataset(
            text,
            Config.block_size
        )

        # -----------------------------
        # Build Model
        # -----------------------------

        self.model = GPT(
            vocab_size=self.dataset.vocab_size,
            embedding_dim=Config.embedding_dim,
            num_heads=Config.num_heads,
            num_layers=Config.num_layers,
            forward_expansion=Config.forward_expansion,
            max_length=Config.block_size
        )

        # -----------------------------
        # Load Best Model
        # -----------------------------

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device,
                weights_only=True
            )
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()


    def generate(
        self,
        prompt,
        max_new_tokens=200,
        temperature=0.7,
        top_k=20
    ):

        # -----------------------------
        # Encode Prompt
        # -----------------------------

        encoded = [
            self.dataset.stoi[c]
            for c in prompt
        ]

        x = torch.tensor(
            encoded,
            dtype=torch.long
        ).unsqueeze(0).to(
            self.device
        )

        # -----------------------------
        # Generate
        # -----------------------------

        with torch.no_grad():

            generated = self.model.generate(
                x,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )

        # -----------------------------
        # Decode
        # -----------------------------

        output = "".join(
            self.dataset.itos[token.item()]
            for token in generated[0]
        )

        return output