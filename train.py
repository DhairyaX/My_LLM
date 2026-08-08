import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from config import Config
from data.dataset import ShakespeareDataset
from models.gpt import GPT


# -----------------------------
# Settings
# -----------------------------

RESUME = False

CHECKPOINT_PATH = "checkpoints/gpt_checkpoint.pth"


# -----------------------------
# Device
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)


# -----------------------------
# Load Dataset
# -----------------------------

with open(
    "data/tiny_shakespeare.txt",
    "r",
    encoding="utf-8"
) as f:
    text = f.read()


full_dataset = ShakespeareDataset(
    text,
    Config.block_size
)


# -----------------------------
# Train / Validation Split
# -----------------------------

train_dataset = Subset(
    full_dataset,
    range(Config.train_size)
)

val_dataset = Subset(
    full_dataset,
    range(
        Config.train_size,
        Config.train_size + Config.val_size
    )
)


# -----------------------------
# DataLoaders
# -----------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=Config.batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=Config.batch_size,
    shuffle=False
)

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))


# -----------------------------
# Build Model
# -----------------------------

model = GPT(
    vocab_size=full_dataset.vocab_size,
    embedding_dim=Config.embedding_dim,
    num_heads=Config.num_heads,
    num_layers=Config.num_layers,
    forward_expansion=Config.forward_expansion,
    max_length=Config.block_size
)

model = model.to(device)


# -----------------------------
# Loss & Optimizer
# -----------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=Config.learning_rate
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=Config.epochs
)


# -----------------------------
# Resume Checkpoint
# -----------------------------

start_epoch = 0

if RESUME and os.path.exists(CHECKPOINT_PATH):

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    start_epoch = checkpoint["epoch"]

    print(
        f"Resuming training from epoch {start_epoch}"
    )


# -----------------------------
# Training Loop
# -----------------------------

best_val_loss = float("inf")

for epoch in range(
    start_epoch,
    Config.epochs
):

    model.train()

    total_train_loss = 0

    for batch_idx, (x, y) in enumerate(train_loader):

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(
            logits.view(-1, full_dataset.vocab_size),
            y.view(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
       )
        optimizer.step()

        total_train_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:

            print(
                f"Epoch {epoch + 1} | "
                f"Batch {batch_idx + 1} | "
                f"Training Loss {loss.item():.4f}"
            )


    # -----------------------------
    # Average Training Loss
    # -----------------------------

    avg_train_loss = (
        total_train_loss / len(train_loader)
    )


    # -----------------------------
    # Validation
    # -----------------------------

    model.eval()

    total_val_loss = 0

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = criterion(
                logits.view(-1, full_dataset.vocab_size),
                y.view(-1)
            )

            total_val_loss += loss.item()


    avg_val_loss = (
        total_val_loss / len(val_loader)
    )
    
    perplexity = torch.exp(
    torch.tensor(avg_val_loss)
    )
    
    if avg_val_loss < best_val_loss:

        best_val_loss = avg_val_loss

        torch.save(
            model.state_dict(),
            "checkpoints/best_model.pth"
        )

        print("New best model saved!")

    # -----------------------------
    # Learning Rate Scheduler
    # -----------------------------

    scheduler.step()


    # -----------------------------
    # Print Results
    # -----------------------------

    print()

    print(
        f"Epoch {epoch + 1} Complete"
    )

    print(
        f"Training Loss:   {avg_train_loss:.4f}"
    )

    print(
        f"Validation Loss: {avg_val_loss:.4f}"
    )
    
    print(
    f"Perplexity:      {perplexity.item():.4f}"
    )
    
    print(
        f"Learning Rate:   "
        f"{optimizer.param_groups[0]['lr']:.6f}"
    )

    print("-" * 50)


    # -----------------------------
    # Save Checkpoint
    # -----------------------------

    os.makedirs(
        "checkpoints",
        exist_ok=True
    )

    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
    }

    torch.save(
        checkpoint,
        CHECKPOINT_PATH
    )

    torch.save(
        model.state_dict(),
        "checkpoints/gpt_model.pth"
    )

    print("Checkpoint saved.")


print("\nTraining finished!")