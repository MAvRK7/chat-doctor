import torch
from tokenizers import Tokenizer
from src.model.transformer import MoETransformer

if __name__ == "__main__":
    tok = Tokenizer.from_file("tokenizer.json")
    vocab_size = tok.get_vocab_size()

    model = MoETransformer(vocab_size=vocab_size)
    x = torch.randint(0, vocab_size, (2, 64))  # (batch, seq)
    logits, moe_loss = model(x)

    print("Logits shape:", logits.shape)
    print("MoE loss:", moe_loss.item())

# Run
# python -m src.model.test_model