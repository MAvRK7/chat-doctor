import torch
import re
import sentencepiece as spm
from src.model.transformer import MoETransformer
from src.sampling import sample, apply_repetition_penalty

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD TOKENIZER
# -----------------------------
sp = spm.SentencePieceProcessor()
sp.load("tokenizer/tokenizer.json.model")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = MoETransformer(
    vocab_size=sp.get_piece_size(),
    dim=512,
    num_layers=8,
    num_heads=8,
    ffn_hidden_dim=2048,
    num_experts=4,
    k=2,
    max_seq_len=1024,
)

state = torch.load("model.pt", map_location=device)
model.load_state_dict(state["model"])
model.to(device)
model.eval()


# -----------------------------
# GENERATION
# -----------------------------
def generate(user_input, max_new_tokens=150, temperature=0.7, top_k=40, top_p=0.9):
    eos_id = sp.eos_id()

    # ✅ FORCE correct format (match your dataset)
    prompt = f"<user> {user_input}\n<assistant> "

    # ✅ Add BOS
    ids_list = [sp.bos_id()] + sp.encode(prompt)
    ids = torch.tensor([ids_list], dtype=torch.long).to(device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _ = model(ids)
            logits = logits[:, -1, :]

            logits = apply_repetition_penalty(
                logits,
                ids[0].tolist(),
                penalty=1.15
            )

            next_id = sample(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

        # stop on EOS
        if eos_id is not None and next_id.item() == eos_id:
            break

        ids = torch.cat([ids, next_id], dim=1)

        # anti-loop
        if len(ids[0]) > 20:
            recent = ids[0][-10:].tolist()
            if len(set(recent)) < 3:
                break

    # -----------------------------
    # DECODE
    # -----------------------------
    text = sp.decode(ids[0].tolist())

    # -----------------------------
    # CLEAN OUTPUT
    # -----------------------------

    # remove prompt
    if text.startswith(prompt):
        text = text[len(prompt):]

    # remove accidental role leakage
    text = re.sub(r"(Patient:|Doctor:)", "", text)

    # remove extra assistant tags if repeated
    if "<assistant>" in text:
        text = text.split("<assistant>", 1)[-1]

    # clean spacing
    text = re.sub(r"\s+", " ", text).strip()

    # remove trailing junk
    text = re.sub(r'["]+$', "", text).strip()

    return text


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    user_input = "My eyes hurt so much, can you suggest what i must do?"  # Example input

    response = generate(user_input)

    print(f"User: {user_input}")
    print(f"Assistant: {response}")

# Run 
# python -m src.inference
