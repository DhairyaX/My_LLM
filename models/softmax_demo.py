import torch
import torch.nn.functional as F

# Example logits (raw scores from the model)
logits = torch.tensor([2.5, 5.1, 0.4, 1.2, 3.8])

print("Logits:")
print(logits)

# Apply Softmax
probabilities = F.softmax(logits, dim=0)

print("\nProbabilities:")
print(probabilities)

print("\nSum of probabilities:")
print(probabilities.sum())

vocab = [
    "i",
    "love",
    "artificial",
    "intelligence",
    "machine"
]

predicted_index = torch.argmax(probabilities)

print("\nPredicted Token Index:")
print(predicted_index.item())

print("\nPredicted Word:")
print(vocab[predicted_index])