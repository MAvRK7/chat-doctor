import math
import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch import amp

from tokenizers import Tokenizer

from src.dataset.dataset import ConversationDataset, collate_batch
from src.model.transformer import MoETransformer


# -----------------------------
# Training Config
# -----------------------------
class TrainConfig:
    dataset_path = "data/processed/cleaned_anon.jsonl"
    tokenizer_path = "tokenizer.json"

    grad_accum_steps = 4
    batch_size = 8
    max_length = 256
    lr = 3e-4
    weight_decay = 0.1
    warmup_steps = 200
    max_steps = 30000

    log_every = 50
    save_every = 1000
    # save_path = "checkpoints/model.pt"
    if "kaggle" in os.getcwd().lower():
        save_path = "/kaggle/working/checkpoints/model.pt"
    else:
        save_path = "/content/checkpoints/model.pt"
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
    ckpt_dir = os.path.dirname(cfg.save_path)
    os.makedirs(ckpt_dir, exist_ok=True)

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

    scaler = amp.GradScaler(device_type="cuda")

    # Make sure gradients are zeroed before training starts
    opt.zero_grad()  

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
            with amp.autocast(device_type="cuda"):
                logits, moe_loss = model(batch)

                logits = logits[:, :-1].contiguous().view(-1, vocab_size)
                labels = labels[:, 1:].contiguous().view(-1)

                ce_loss = ce_loss_fn(logits, labels)
                loss = ce_loss + 0.01 * moe_loss


            # Backward
            loss = loss / cfg.grad_accum_steps
            scaler.scale(loss).backward()

            if (step + 1) % cfg.grad_accum_steps == 0:
                scaler.unscale_(opt)
                clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()


            # Update LR
            lr = cosine_lr(step, cfg.max_steps, cfg.lr, cfg.warmup_steps)
            for g in opt.param_groups:
                g["lr"] = lr


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