import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch import amp
from functools import partial

from tokenizers import Tokenizer
from torch.utils.tensorboard import SummaryWriter

from src.dataset.dataset import ConversationDataset, collate_batch
from src.model.transformer import MoETransformer


# -----------------------------
# SPEED SETTINGS
# -----------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# -----------------------------
# SAFE COLLATE (PICKLABLE)
# -----------------------------
def collate_fn_wrapper(batch):
    return collate_batch(batch, pad_id=0)


# -----------------------------
# Config
# -----------------------------
class TrainConfig:
    repo_root = os.getcwd()

    train_path = os.path.join(repo_root, "data/processed/train.jsonl")
    val_path = os.path.join(repo_root, "data/processed/val.jsonl")
    tokenizer_path = os.path.join(repo_root, "tokenizer.json")
    save_path = os.path.join(repo_root, "checkpoints/model.pt")

    log_dir = os.path.join(repo_root, "outputs", "runs")

    grad_accum_steps = 2
    batch_size = 12
    max_length = 512

    lr = 3e-4
    weight_decay = 0.1
    warmup_steps = 200
    max_steps = 30000

    log_every = 50
    eval_every = 1000

    min_delta = 1e-4
    patience = 5

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 🔥 Important for Mac vs Kaggle
    num_workers = 0 if device == "cpu" else 2


# -----------------------------
# LR Scheduler
# -----------------------------
def cosine_lr(step, max_steps, base_lr, warmup_steps):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


# -----------------------------
# Eval
# -----------------------------
def evaluate(model, dl, device, vocab_size):
    model.eval()
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    total_loss = 0
    count = 0

    with torch.no_grad(), amp.autocast(device_type="cuda", enabled=(device == "cuda")):
        for batch, labels in dl:
            batch = batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            attention_mask = (batch != 0)

            logits, _ = model(batch, attention_mask=attention_mask)

            logits = logits[:, :-1].contiguous().view(-1, vocab_size)
            labels = labels[:, 1:].contiguous().view(-1)

            loss = ce_loss_fn(logits, labels)
            total_loss += loss.item() * labels.numel()
            count += labels.numel()

    model.train()
    avg = total_loss / max(count, 1)
    ppl = math.exp(min(avg, 20))
    return avg, ppl


# -----------------------------
# Train
# -----------------------------
def train():
    cfg = TrainConfig()

    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=cfg.log_dir)

    tok = Tokenizer.from_file(cfg.tokenizer_path)
    vocab_size = tok.get_vocab_size()

    train_ds = ConversationDataset(cfg.train_path, tok, cfg.max_length)
    val_ds = ConversationDataset(cfg.val_path, tok, cfg.max_length)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn_wrapper,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn_wrapper,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

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

    # 🔥 SPEED BOOST
    if cfg.device == "cuda":
        model = torch.compile(model)

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = amp.GradScaler(enabled=(cfg.device == "cuda"))

    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    # -----------------------------
    # Resume if checkpoint exists
    # -----------------------------
    if os.path.exists(cfg.save_path):
        print("Resuming from checkpoint:", cfg.save_path)
        ckpt = torch.load(cfg.save_path, map_location=cfg.device)

        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        step = ckpt["step"]

        # Move optimizer tensors to correct device
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(cfg.device)

        print(f"Resumed at step {step}")
    else:
        step = 0

    opt.zero_grad()

    best_val = float("inf")
    no_improve = 0

    model.train()

    while step < cfg.max_steps:
        for batch, labels in train_dl:

            batch = batch.to(cfg.device, non_blocking=True)
            labels = labels.to(cfg.device, non_blocking=True)
            attention_mask = (batch != 0)

            with amp.autocast(device_type="cuda", enabled=(cfg.device == "cuda")):
                logits, aux = model(batch, attention_mask=attention_mask)

                logits_flat = logits[:, :-1].contiguous().view(-1, vocab_size)
                labels_flat = labels[:, 1:].contiguous().view(-1)

                ce_loss = ce_loss_fn(logits_flat, labels_flat)

                if torch.is_tensor(aux):
                    moe_loss = aux
                elif isinstance(aux, dict):
                    moe_loss = aux.get("moe_loss", torch.tensor(0.0, device=ce_loss.device))
                else:
                    moe_loss = torch.tensor(0.0, device=ce_loss.device)

                total_loss = (ce_loss + 0.01 * moe_loss) / cfg.grad_accum_steps

            scaler.scale(total_loss).backward()

            if (step + 1) % cfg.grad_accum_steps == 0:
                lr = cosine_lr(step, cfg.max_steps, cfg.lr, cfg.warmup_steps)

                for g in opt.param_groups:
                    g["lr"] = lr

                scaler.unscale_(opt)
                clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

                writer.add_scalar("train/lr", lr, step)

            if step % cfg.log_every == 0:
                current_loss = (ce_loss + 0.01 * moe_loss).detach().item()

                writer.add_scalar("train/loss", current_loss, step)
                writer.add_scalar("train/ce_loss", ce_loss.item(), step)
                writer.add_scalar("train/moe_loss", moe_loss.detach().item(), step)
                writer.flush()

                print(
                    f"Step {step} | "
                    f"Loss: {current_loss:.4f} | "
                    f"CE: {ce_loss.item():.4f} | "
                    f"MoE: {moe_loss.detach().item():.4f}"
                )

            if step % cfg.eval_every == 0 and step > 0:
                val_loss, val_ppl = evaluate(model, val_dl, cfg.device, vocab_size)

                writer.add_scalar("val/loss", val_loss, step)
                writer.add_scalar("val/perplexity", val_ppl, step)
                writer.flush()

                print(f"[VAL] {step} | loss={val_loss:.4f} | ppl={val_ppl:.2f}")

                if val_loss < best_val - cfg.min_delta:
                    best_val = val_loss
                    no_improve = 0

                    torch.save(model.state_dict(), cfg.save_path)
                    print("[BEST SAVED]")
                else:
                    no_improve += 1
                    if no_improve >= cfg.patience:
                        print("Early stopping")
                        writer.close()
                        return

            step += 1
            if step >= cfg.max_steps:
                break

    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
    }, cfg.save_path)

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    train()

# run 
# python -m src.train