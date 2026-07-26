from tokenizer.simple_tokenizer import SimpleTokenizer
from data.prepare_data import words

# Build vocabulary
vocab = sorted(set(words))

# Initialize tokenizer
tokenizer = SimpleTokenizer(vocab)

print("Vocabulary:")
print(vocab)
print()

print("Encoded Dataset:\n")

for i in range(len(words) - 1):
    input_words = words[:i + 1]
    target_word = words[i + 1]

    input_text = " ".join(input_words)

    input_ids = tokenizer.encode(input_text)
    target_id = tokenizer.encode(target_word)[0]

    print(f"Input Words : {input_words}")
    print(f"Input IDs   : {input_ids}")
    print(f"Target Word : {target_word}")
    print(f"Target ID   : {target_id}")
    print("-" * 50)