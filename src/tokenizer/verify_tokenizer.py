import sentencepiece as spm

def verify_tokenizer(model_path="src/tokenizer/tokenizer.json.model"):
    print("Loading SentencePiece tokenizer...")
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)  # Load the SentencePiece model

    # Check vocab size
    vocab_size = sp.get_piece_size()
    print(f"Vocab size: {vocab_size}")

    # Test encoding
    test_text = "Explain what a migraine is."
    enc = sp.encode(test_text, out_type=int)  # Encoding as integer token IDs
    print("\nTest text:", test_text)
    print("Encoded IDs:", enc)
    print("Decoded text:", sp.decode(enc))  # Decoding back to text

    # Check special tokens
    print("\nSpecial tokens:")
    for token in ["<pad>", "<bos>", "<eos>", "<unk>"]:
        token_id = sp.piece_to_id(token)  # Get token ID for special tokens
        if token_id != -1:
            print(f"{token}: {token_id}")
        else:
            print(f"{token}: NOT FOUND")

    # Basic evaluation: Tokenization consistency
    # Evaluate the tokenizer with a small corpus of test sentences
    test_sentences = [
        "What is the capital of France?",
        "How does the immune system work?",
        "Describe the process of digestion."
    ]
    
    print("\nEvaluating tokenization on sample sentences:")
    for sentence in test_sentences:
        enc = sp.encode(sentence, out_type=int)
        print(f"\nSentence: {sentence}")
        print(f"Encoded IDs: {enc}")
        print(f"Decoded Text: {sp.decode(enc)}")

if __name__ == "__main__":
    verify_tokenizer()


# Rin this
# python src/tokenizer/verify_tokenizer.py