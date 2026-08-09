class Config:

    # -----------------------------
    # Dataset
    # -----------------------------

    block_size = 128

    # -----------------------------
    # Model
    # -----------------------------

    embedding_dim = 128

    num_heads = 4

    num_layers = 4

    forward_expansion = 4

    dropout = 0.1

    # -----------------------------
    # Training
    # -----------------------------

    batch_size = 32

    learning_rate = 3e-4

    epochs = 1

    # -----------------------------
    # Device
    # -----------------------------

    device = "cuda"