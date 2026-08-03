import torch

# Two vectors
a = torch.tensor([1.0, 2.0, 3.0])

b = torch.tensor([4.0, 1.0, 2.0])

dot = torch.dot(a, b)

print("Vector A:", a)
print("Vector B:", b)

print("\nDot Product:", dot)