import math
import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from tokenizers import Tokenizer

from src.dataset.dataset import ConversationDataset, collate_batch
from src.model.transformer import MoETransformer


# -----------------------------
# Training Config
# -----------------------------
class TrainConfig:
    dataset_path = "data/processed/cleaned_anon.jsonl"
    tokenizer_path = "tokenizer.json"

    batch_size = 8
    max_length = 256
    lr = 3e-4
    weight_decay = 0.1
    warmup_steps = 200
    max_steps = 30000

    log_every = 50
    save_every = 1000
    # save_path = "checkpoints/model.pt"
    save_path = "/kaggle/working/checkpoints/model.pt"
    resume = True

    device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Learning Rate Scheduler (Cosine)
# -----------------------------
def cosine_lr(step, max_steps, base_lr, warmup_steps):
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


# -----------------------------
# Training Loop
# -----------------------------
def train():
    cfg = TrainConfig()

    # Ensure checkpoint directory exists
    # os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
    os.makedirs("/kaggle/working/checkpoints", exist_ok=True)

    # Load tokenizer
    tok = Tokenizer.from_file(cfg.tokenizer_path)
    vocab_size = tok.get_vocab_size()

    # Dataset + DataLoader
    ds = ConversationDataset(
        path=cfg.dataset_path,
        tokenizer=tok,
        max_length=cfg.max_length,
    )

    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_batch(x, pad_id=0),
    )

    # Model
    model = MoETransformer(
        vocab_size=vocab_size,
        dim=512,
        num_layers=8,
        num_heads=8,
        ffn_hidden_dim=2048,
        num_experts=4,
        k=2,
        max_seq_len=cfg.max_length,
    ).to(cfg.device)

    # Optimizer
    opt = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # resume from checkpoint
    start_step = 0

    if cfg.resume:
        if os.path.exists(cfg.save_path):
            try:
                print(f"Resuming from {cfg.save_path}")
                checkpoint = torch.load(cfg.save_path, map_location=cfg.device)
                model.load_state_dict(checkpoint["model"])
                opt.load_state_dict(checkpoint["optimizer"])
                start_step = checkpoint["step"]
                print(f"Successfully resumed from step {start_step}")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                print("Starting from step 0 instead.")
        else:
            print("No checkpoint found. Starting from step 0.")


    # Loss
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    step = start_step
    model.train()

    for epoch in range(999999):  # effectively infinite until max_steps
        for batch, labels in dl:
            batch = batch.to(cfg.device)
            labels = labels.to(cfg.device)

            # Forward
            logits, moe_loss = model(batch)

            # Shift for next-token prediction
            logits = logits[:, :-1].contiguous().view(-1, vocab_size)
            labels = labels[:, 1:].contiguous().view(-1)

            ce_loss = ce_loss_fn(logits, labels)

            # Total loss = CE + MoE aux loss
            loss = ce_loss + 0.01 * moe_loss

            # Backward
            opt.zero_grad()
            loss.backward()

            # Gradient clipping
            clip_grad_norm_(model.parameters(), 1.0)

            # Update LR
            lr = cosine_lr(step, cfg.max_steps, cfg.lr, cfg.warmup_steps)
            for g in opt.param_groups:
                g["lr"] = lr

            opt.step()

            # Logging
            if step % cfg.log_every == 0:
                print(
                    f"Step {step} | "
                    f"Loss: {loss.item():.4f} | "
                    f"CE: {ce_loss.item():.4f} | "
                    f"MoE: {moe_loss.item():.4f} | "
                    f"LR: {lr:.6f}"
                )

            # Save checkpoint (including optimizer and step)
            if step % cfg.save_every == 0 and step > 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "step": step,
                    },
                    cfg.save_path,
                )
                print(f"Saved checkpoint to {cfg.save_path}")

            step += 1
            if step >= cfg.max_steps:
                print("Training complete.")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "step": step,
                    },
                    cfg.save_path,
                )
                print(f"Saved checkpoint to {cfg.save_path}")
                return


if __name__ == "__main__":
    train()