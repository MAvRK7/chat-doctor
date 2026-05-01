'''
import os
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch import amp
from tokenizers import Tokenizer
from torch.utils.tensorboard import SummaryWriter

from src.model.transformer import MoETransformer


# ============================================================
# CONFIG
# ============================================================

class FTConfig:
    repo_root = os.getcwd()

    train_path = os.path.join(repo_root, "data/finetune/train.jsonl")
    val_path = os.path.join(repo_root, "data/finetune/val.jsonl")
    tokenizer_path = os.path.join(repo_root, "tokenizer.json")

    checkpoint_path = os.path.join(repo_root, "model.pt")
    save_path = os.path.join(repo_root, "model_ft.pt")

    log_dir = os.path.join(repo_root, "outputs", "runs_ft")

    batch_size = 1               # M1 safe
    grad_accum_steps = 8         # effective batch size = 8
    max_length = 512

    lr = 1e-5
    weight_decay = 0.01

    max_steps = 3000
    warmup_steps = 100

    log_every = 50
    eval_every = 300

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


# ============================================================
# LR SCHEDULE
# ============================================================

def cosine_lr(step, max_steps, base_lr, warmup_steps):
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    progress = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, dl, device, vocab_size):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    total_loss = 0
    count = 0

    with torch.no_grad():
        for batch, labels in dl:
            batch = batch.to(device)
            labels = labels.to(device)

            logits, _ = model(batch, compute_moe_loss=False)

            logits = logits[:, :-1].contiguous().view(-1, vocab_size)
            labels = labels[:, 1:].contiguous().view(-1)

            mask = labels != -100
            if mask.sum() == 0:
                continue

            loss = loss_fn(logits, labels)
            total_loss += loss.item() * mask.sum().item()
            count += mask.sum().item()

    model.train()
    return total_loss / max(count, 1)


# ============================================================
# TRAIN LOOP
# ============================================================

def train():
    cfg = FTConfig()

    os.makedirs(cfg.log_dir, exist_ok=True)
    writer = SummaryWriter(cfg.log_dir)

    tok = Tokenizer.from_file(cfg.tokenizer_path)
    vocab_size = tok.get_vocab_size()

    train_ds = InstructionDataset(cfg.train_path, tok, cfg.max_length)
    val_ds = InstructionDataset(cfg.val_path, tok, cfg.max_length)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

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

    checkpoint = torch.load(cfg.checkpoint_path, map_location=cfg.device)
    model.load_state_dict(checkpoint["model"], strict=False)
    print("Loaded base model.")

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = amp.GradScaler(enabled=(cfg.device == "cuda"))
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    step = 0
    opt.zero_grad()
    model.train()

    while step < cfg.max_steps:
        for batch, labels in train_dl:
            batch = batch.to(cfg.device)
            labels = labels.to(cfg.device)

            with amp.autocast(device_type="cuda", enabled=(cfg.device == "cuda")):
                logits, _ = model(batch, compute_moe_loss=False)

                logits = logits[:, :-1].contiguous().view(-1, vocab_size)
                labels_flat = labels[:, 1:].contiguous().view(-1)

                loss = loss_fn(logits, labels_flat) / cfg.grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % cfg.grad_accum_steps == 0:
                lr = cosine_lr(step, cfg.max_steps, cfg.lr, cfg.warmup_steps)
                for g in opt.param_groups:
                    g["lr"] = lr

                scaler.unscale_(opt)
                clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/lr", lr, step)

                print(f"Step {step} | Loss {loss.item():.4f}")

            if step % cfg.eval_every == 0 and step > 0:
                val_loss = evaluate(model, val_dl, cfg.device, vocab_size)
                print(f"[VAL] {step} | loss={val_loss:.4f}")
                writer.add_scalar("val/loss", val_loss, step)

                torch.save({"model": model.state_dict()}, cfg.save_path)
                print("Checkpoint saved.")

            step += 1
            if step >= cfg.max_steps:
                break

    torch.save({"model": model.state_dict()}, cfg.save_path)
    print("Fine-tuning complete.")


if __name__ == "__main__":
    train()

# run 
# python -m src.train_ft
'''