import torch
import re
from tokenizers import Tokenizer
from src.model.transformer import MoETransformer
from src.sampling import sample, apply_repetition_penalty

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")

# Load model
model = MoETransformer(
    vocab_size=tokenizer.get_vocab_size(),
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

def generate(prompt, max_new_tokens=150, temperature=0.7, top_k=40, top_p=0.9):
    eos_id = tokenizer.token_to_id("<eos>")

    enc = tokenizer.encode(prompt)
    ids = torch.tensor([enc.ids], dtype=torch.long).to(device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _ = model(ids)
            logits = logits[:, -1, :]

            # repetition penalty
            logits = apply_repetition_penalty(
                logits,
                ids[0].tolist(),
                penalty=1.15
            )

            # sampling
            next_id = sample(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

        # stop if EOS
        if eos_id is not None and next_id.item() == eos_id:
            break

        ids = torch.cat([ids, next_id], dim=1)

    # decode
    text = tokenizer.decode(ids[0].tolist())

    # clean byte-level artifacts
    text = text.replace("Ġ", " ").replace("Ċ", "\n")

    # collapse only *double* spaces, not all whitespace
    text = re.sub(r" {2,}", " ", text)

    # remove leftover placeholders
    text = text.replace("Person", "").strip()

    # stop at first "Assistant:" or repeated role
    text = re.split(r"(Assistant:|Doctor: Patient:)", text)[0].strip()

    return text

if __name__ == "__main__":
    prompt = (
        "Patient: I have a headache."
        "Doctor:"
    )
    print(generate(prompt))

# Run 
# python -m src.inference
