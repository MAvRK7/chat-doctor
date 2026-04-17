from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors

def train_tokenizer(input_file, vocab_size=8000, save_path="tokenizer.json"):
    # Create BPE tokenizer, <unk> for unknown tokens
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    # Splits text at the byte level— break everything down into raw bytes first.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    # Learns up to vocab_size tokens (default: 8000), including special tokens
    # pad, beg of sentance, end of sentance, unknown
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"]
    )

    # train the data
    tokenizer.train([input_file], trainer)

    # Post processing
    # Single input: <bos> your_text <eos> Pair input: <bos> text1 <eos> text2 <eos>
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> $B:1 <eos>:1",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )

    # save tokenizer to disk
    tokenizer.save(save_path)
    print(f"✅ Tokenizer saved to {save_path}")

# comand line usage
# python src/tokenizer/train_tokenizer.py --input data/processed/cleaned_anon.jsonl --vocab_size 8000 --output tokenizer.json

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=8000)
    args = parser.parse_args()

    train_tokenizer(args.input, args.vocab_size, args.output)
