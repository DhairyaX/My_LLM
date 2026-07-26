from prepare_data import words

print("Words:")
print(words)
print()

for i in range(len(words) - 1):
    input_sequence = words[:i + 1]
    target = words[i + 1]

    print(f"Input : {input_sequence}")
    print(f"Target: {target}")
    print("-" * 40)