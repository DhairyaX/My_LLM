class Config:

    # Dataset
    vocab_size = 65
    block_size = 128

    # Model
    embedding_dim = 128
    num_heads = 4
    num_layers = 4
    forward_expansion = 4

    # Training
    batch_size = 32
    learning_rate = 3e-4
    epochs = 4

    # Device
    device = "cuda"
    
    train_size = 4000
    val_size = 1000
    
    dropout = 0.1