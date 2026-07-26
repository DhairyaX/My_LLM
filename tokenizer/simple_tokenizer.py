class SimpleTokenizer:
    def __init__(self, vocab):
        """
        Initialize the tokenizer with a vocabulary.

        Args:
            vocab (list): List of unique words.
        """
        self.vocab = vocab

        # Create word -> id mapping
        self.word_to_id = {
            word: idx for idx, word in enumerate(vocab)
        }

        # Create id -> word mapping
        self.id_to_word = {
            idx: word for idx, word in enumerate(vocab)
        }

    def encode(self, text):
        """
        Convert text into token IDs.

        Args:
            text (str): Input sentence.

        Returns:
            list: List of token IDs.
        """
        words = text.split()

        token_ids = []

        for word in words:
            if word not in self.word_to_id:
                raise ValueError(f"Unknown word: '{word}'")

            token_ids.append(self.word_to_id[word])

        return token_ids

    def decode(self, token_ids):
        """
        Convert token IDs back to text.

        Args:
            token_ids (list): List of token IDs.

        Returns:
            str: Decoded sentence.
        """
        words = []

        for token_id in token_ids:
            if token_id not in self.id_to_word:
                raise ValueError(f"Unknown token ID: {token_id}")

            words.append(self.id_to_word[token_id])

        return " ".join(words)