import json
import sentencepiece as spm
from tqdm import tqdm

def count_tokens(tokenizer_path, dataset_path):
    # Load the SentencePiece model
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)

    total_tokens = 0
    num_samples = 0

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Counting tokens"):
            obj = json.loads(line)
            msgs = obj["messages"]

            # Combine messages into one training string
            text = ""
            for m in msgs:
                prefix = "User: " if m["role"] == "user" else "Assistant: "
                text += prefix + m["content"].strip() + "\n"

            # Tokenize the text using the SentencePiece model
            enc = sp.encode(text, out_type=int)
            total_tokens += len(enc)
            num_samples += 1

    print(f"\nTotal samples: {num_samples}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average tokens per sample: {total_tokens / num_samples:.2f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to the tokenizer model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset")
    args = parser.parse_args()

    count_tokens(args.tokenizer, args.dataset)


# Run 
# python src/tokenizer/count_tokens.py --tokenizer src/tokenizer/tokenizer.json.model --dataset data/processed/train.jsonl