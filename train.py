from tokenizer.simple_tokenizer import SimpleTokenizer

vocab = ["I", "love", "AI", "cats"]

tokenizer = SimpleTokenizer(vocab)

encoded = tokenizer.encode("I love Chatgpt")
print("Encoded:", encoded)

# decoded = tokenizer.decode(encoded)
# print("Decoded:", decoded)