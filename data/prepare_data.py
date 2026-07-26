import string

# Sample training text
text = """
I love artificial intelligence.
Artificial intelligence is changing the world.
I love machine learning.
Machine learning is amazing.
"""

# Step 1: Convert to lowercase
text = text.lower()

# Step 2: Remove punctuation
translator = str.maketrans("", "", string.punctuation)
text = text.translate(translator)

# Step 3: Split text into words
words = text.split()

# Step 4: Build the vocabulary
vocab = sorted(set(words))

# Step 5: Print results
print("Words:")
print(words)

print("\nTotal words:", len(words))

print("\nVocabulary:")
print(vocab)

print("\nVocabulary Size:", len(vocab))