import torch

seq_length = 5

mask = torch.tril(torch.ones(seq_length, seq_length))

print(mask)