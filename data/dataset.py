import torch
from torch.utils.data import Dataset


class ShakespeareDataset(Dataset):

    def __init__(self, text, block_size):

        chars = sorted(list(set(text)))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        self.vocab_size = len(chars)

        self.data = torch.tensor(
            [self.stoi[c] for c in text],
            dtype=torch.long
        )

        self.block_size = block_size

    def __len__(self):

        return len(self.data) - self.block_size

    def __getitem__(self, idx):

        x = self.data[idx:idx+self.block_size]

        y = self.data[idx+1:idx+self.block_size+1]

        return x, y