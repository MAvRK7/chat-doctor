from tokenizers import Tokenizer

def verify_tokenizer(path="tokenizer.json"):
    print("Loading tokenizer...")
    tok = Tokenizer.from_file(path)

    # Check vocab size
    vocab_size = tok.get_vocab_size()
    print(f"Vocab size: {vocab_size}")

    # Test encoding
    test_text = "Explain what a migraine is."
    enc = tok.encode(test_text)
    print("\nTest text:", test_text)
    print("Encoded IDs:", enc.ids)
    print("Decoded text:", tok.decode(enc.ids))

    # Check special tokens
    print("\nSpecial tokens:")
    for token in ["<pad>", "<bos>", "<eos>", "<unk>"]:
        try:
            print(f"{token}: {tok.token_to_id(token)}")
        except Exception:
            print(f"{token}: NOT FOUND")

if __name__ == "__main__":
    verify_tokenizer()

# Rin this
# python src/tokenizer/verify_tokenizer.py