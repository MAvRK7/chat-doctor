import math
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch import amp

from tokenizers import Tokenizer
from torch.utils.tensorboard import SummaryWriter

from src.dataset.dataset import ConversationDataset, collate_batch
from src.model.transformer import MoETransformer


# -----------------------------
# Config
# -----------------------------
class TrainConfig:
    repo_root = os.getcwd()  # IMPORTANT for Kaggle

    train_path = os.path.join(repo_root, "data/processed/train.jsonl")
    val_path = os.path.join(repo_root, "data/processed/val.jsonl")
    tokenizer_path = os.path.join(repo_root, "tokenizer.json")
    save_path = os.path.join(repo_root, "checkpoints/model.pt")

    log_dir = os.path.join(repo_root, "outputs", "runs")

    grad_accum_steps = 4
    batch_size = 8
    max_length = 512
    lr = 3e-4
    weight_decay = 0.1
    warmup_steps = 200
    max_steps = 30000

    log_every = 50
    eval_every = 500

    min_delta = 1e-4
    patience = 5

    device = "cuda" if torch.cuda.is_available() else "cpu"


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
            attention_mask = (batch != 0)

            batch = batch.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

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
# Generation
# -----------------------------
def generate_text(model, tok, prompt, device, max_new_tokens=80):
    model.eval()

    with torch.no_grad(), amp.autocast(device_type="cuda", enabled=(device == "cuda")):
        ids = tok.encode(prompt).ids
        input_ids = torch.tensor([ids], device=device)

        for _ in range(max_new_tokens):
            logits, _ = model(input_ids)
            probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            input_ids = input_ids[:, -model.max_seq_len:]

    out = tok.decode(input_ids[0].tolist())
    model.train()
    return out


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
        collate_fn=lambda x: collate_batch(x, pad_id=0),
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_batch(x, pad_id=0),
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

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = amp.GradScaler(enabled=(cfg.device == "cuda"))

    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    step = 0
    opt.zero_grad()

    best_val = float("inf")
    no_improve = 0
    best_checkpoint = None

    opt_step = 0

    model.train()

    while step < cfg.max_steps:
        for batch, labels in train_dl:
            attention_mask = (batch != 0)

            batch = batch.to(cfg.device)
            labels = labels.to(cfg.device)
            attention_mask = attention_mask.to(cfg.device)

            with amp.autocast(device_type="cuda", enabled=(cfg.device == "cuda")):
                logits, aux = model(batch, attention_mask=attention_mask)

                logits_flat = logits[:, :-1].contiguous().view(-1, vocab_size)
                labels_flat = labels[:, 1:].contiguous().view(-1)

                ce_loss = ce_loss_fn(logits_flat, labels_flat)

                moe_loss = aux if torch.is_tensor(aux) else aux.get("moe_loss", 0.0)

                total_loss = (ce_loss + 0.01 * moe_loss) / cfg.grad_accum_steps

            scaler.scale(total_loss).backward()

            if (step + 1) % cfg.grad_accum_steps == 0:
                opt_step += 1

                lr = cosine_lr(opt_step, cfg.max_steps, cfg.lr, cfg.warmup_steps)
                for g in opt.param_groups:
                    g["lr"] = lr

                scaler.unscale_(opt)
                clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

                writer.add_scalar("train/lr", lr, opt_step)

            if step % cfg.log_every == 0:
                current_loss = (ce_loss + 0.01 * moe_loss).detach().item()

                writer.add_scalar("train/loss", current_loss, step)
                writer.add_scalar("train/ce_loss", ce_loss.item(), step)
                writer.add_scalar("train/moe_loss", moe_loss.detach().item(), step)

                print(
                    f"Step {step} | "
                    f"Loss: {current_loss:.4f} | "
                    f"CE: {ce_loss.item():.4f} | "
                    f"MoE: {moe_loss.detach().item():.4f}"
                )

            # -----------------------------
            # Evaluation
            # -----------------------------
            if step % cfg.eval_every == 0 and step > 0:
                val_loss, val_ppl = evaluate(model, val_dl, cfg.device, vocab_size)

                writer.add_scalar("val/loss", val_loss, step)
                writer.add_scalar("val/perplexity", val_ppl, step)

                print(f"[VAL] {step} | loss={val_loss:.4f} | ppl={val_ppl:.2f}")

                sample = generate_text(
                    model,
                    tok,
                    "Patient: I have a headache.\nDoctor:",
                    cfg.device,
                )

                writer.add_text("samples/output", sample, step)
                print("[GEN]", sample)

                # ---- EARLY STOPPING ----
                if val_loss < best_val - cfg.min_delta:
                    best_val = val_loss
                    no_improve = 0

                    best_checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "step": step,
                    }

                    torch.save(best_checkpoint, cfg.save_path)
                    print("[BEST SAVED]")
                else:
                    no_improve += 1
                    print(f"No improve: {no_improve}/{cfg.patience}")

                    if no_improve >= cfg.patience:
                        print("Early stopping. Restoring best model...")
                        if best_checkpoint is not None:
                            model.load_state_dict(best_checkpoint["model"])
                        writer.close()
                        return

            step += 1
            if step >= cfg.max_steps:
                break

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "step": step,
        },
        cfg.save_path,
    )

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    train()

# run 
# python -m src.train