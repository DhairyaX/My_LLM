from data.dataset import ShakespeareDataset

from config import Config

with open(
    "data/tiny_shakespeare.txt",
    "r",
    encoding="utf-8"
) as f:

    text = f.read()

dataset = ShakespeareDataset(
    text,
    Config.block_size
)

print("Vocabulary Size:", dataset.vocab_size)

print("Dataset Length:", len(dataset))

x, y = dataset[0]

print()

print("Input Shape :", x.shape)

print("Target Shape:", y.shape)

print()

print(x)

print()

print(y)