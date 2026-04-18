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
repo_root = os.path.dirname(os.path.abspath(__file__))  # src/
repo_root = os.path.dirname(repo_root)                  # chat-doctor/

class TrainConfig:
    train_path = os.path.join(repo_root, "data/processed/train.jsonl")
    val_path = os.path.join(repo_root, "data/processed/val.jsonl")
    tokenizer_path = os.path.join(repo_root, "tokenizer.json")
    save_path = os.path.join(repo_root, "checkpoints/model.pt")


    grad_accum_steps = 4
    batch_size = 8
    max_length = 512
    lr = 3e-4
    weight_decay = 0.1
    warmup_steps = 200
    max_steps = 30000

    log_every = 50
    save_every = 1000
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
# Evaluation
# -----------------------------

def evaluate(model, dl, device, vocab_size):
    model.eval()
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0
    count = 0
    use_amp = (device == "cuda")

    with torch.no_grad(), amp.autocast(device_type="cuda", enabled=use_amp):
        for batch, labels in dl:
            batch = batch.to(device)
            labels = labels.to(device)

            logits, moe_loss = model(batch)
            logits = logits[:, :-1].contiguous().view(-1, vocab_size)
            labels = labels[:, 1:].contiguous().view(-1)

            ce_loss = ce_loss_fn(logits, labels)
            total_loss += ce_loss.item() * labels.numel()
            count += labels.numel()


    model.train()
    avg = total_loss / max(count, 1)
    ppl = math.exp(min(avg, 20))
    return avg, ppl


# -----------------------------
# Text Generation
# -----------------------------

def generate_text(model, tok, prompt, device, max_new_tokens=80):
    model.eval()
    use_amp = (device == "cuda")
    with torch.no_grad(), amp.autocast(device_type="cuda", enabled=use_amp):
        ids = tok.encode(prompt).ids
        input_ids = torch.tensor([ids], device=device)

        for _ in range(max_new_tokens):
            logits, _ = model(input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            input_ids = input_ids[:, -model.max_seq_len:]

        out = tok.decode(input_ids[0].tolist())
    model.train()
    return out

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
    train_ds = ConversationDataset(
        path=cfg.train_path,
        tokenizer=tok,
        max_length=cfg.max_length,
    )

    val_ds = ConversationDataset(
        path=cfg.val_path,
        tokenizer=tok,
        max_length=cfg.max_length,
    )

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

    scaler = amp.GradScaler(enabled=(cfg.device == "cuda"))

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

    best_val_loss = float("inf")
    no_improve = 0
    patience = 5
    eval_every = 1000

    for epoch in range(999999):  # effectively infinite until max_steps
        for batch, labels in train_dl:
            batch = batch.to(cfg.device)
            labels = labels.to(cfg.device)

            # Forward
            use_amp = (cfg.device == "cuda")
            with amp.autocast(device_type="cuda", enabled=use_amp):
                out = model(batch)
                if len(out) == 3:
                    logits, moe_loss, expert_usage = out
                else:
                    logits, moe_loss = out
                    expert_usage = None


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
            true_loss = ce_loss + 0.01 * moe_loss
            if step % cfg.log_every == 0:
                print(
                    f"Step {step} | "
                    f"Loss: {true_loss.item():.4f} | "
                    f"CE: {ce_loss.item():.4f} | "
                    f"MoE: {moe_loss.item():.4f} | "
                    f"LR: {lr:.6f}"
                )
                if expert_usage is not None:
                    print("Expert usage:", expert_usage.mean(dim=0).detach().cpu().numpy())
            
            # Evaluation
            if step % eval_every == 0 and step > 0:
                val_loss, val_ppl = evaluate(model, val_dl, cfg.device, vocab_size)
                print(f"[VAL] Step {step} | loss={val_loss:.4f} | ppl={val_ppl:.2f}")

                sample = generate_text(
                    model, tok,
                    prompt="Patient: I have a headache.\nDoctor:",
                    device=cfg.device
                )
                print("[GEN]", sample)

                # Early stopping + best model save
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve = 0
                    torch.save(
                        {"model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "step": step},
                        cfg.save_path
                    )
                    print("[BEST] Model saved")
                else:
                    no_improve += 1
                    print(f"[EARLY STOP] No improvement for {no_improve} evals")
                    if no_improve >= patience:
                        print("Early stopping triggered")
                        return


            # Save checkpoint (including optimizer and step)
            '''
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
            '''

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

# Run
# python -m src.train