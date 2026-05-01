import sentencepiece as spm
import json
import argparse
import random
import re
import os
from tqdm import tqdm


# -----------------------------
# OPTIONAL NORMALIZATION
# -----------------------------
def normalize_text(text: str) -> str:
    text = text.strip()

    text = re.sub(r'\s+', ' ', text)  # collapse spaces

    text = re.sub(
        r'(\d+)(mg|ml|kg|g|mcg|mmhg|bpm|cm|mm)',
        r'\1 \2',
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(r'(\d+)/(\d+)', r'\1 / \2', text)

    return text


# -----------------------------
# LOAD JSONL (IMPORTANT FIX)
# -----------------------------
def load_jsonl(path, sample_size=None):
    texts = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc=f"Loading {path}")):
            obj = json.loads(line)

            text = obj.get("text", "")
            if not text:
                continue

            text = normalize_text(text)

            if len(text) < 20:
                continue

            texts.append(text)

            if sample_size and len(texts) >= sample_size:
                break

    return texts


# -----------------------------
# WRITE CORPUS
# -----------------------------
def write_corpus(texts, path):
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n")


# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", type=str, default="tokenizer")

    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--sample_size", type=int, default=None)

    args = parser.parse_args()

    all_texts = []

    print("\n📥 Loading datasets...")
    for path in args.input:
        texts = load_jsonl(path, args.sample_size)
        print(f"Loaded {len(texts)} from {path}")
        all_texts.extend(texts)

    print(f"\nTotal before dedup: {len(all_texts)}")

    all_texts = list(set(all_texts))
    random.shuffle(all_texts)

    print(f"After dedup: {len(all_texts)}")

    os.makedirs("tokenizer", exist_ok=True)

    corpus_path = "tokenizer/corpus.txt"

    print("\n💾 Writing corpus...")
    write_corpus(all_texts, corpus_path)

    print("\n🧠 Training SentencePiece tokenizer...")

    spm.SentencePieceTrainer.Train(
        input=corpus_path,
        model_prefix=f"tokenizer/{args.output}",
        vocab_size=args.vocab_size,
        model_type="bpe",

        character_coverage=0.9995,
        normalization_rule_name="nmt_nfkc",

        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,

        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",

        user_defined_symbols=[
            "<user>",
            "<assistant>",
        ],

        shuffle_input_sentence=True,
        input_sentence_size=5000000
    )

    print("\n✅ Tokenizer saved to tokenizer/")


if __name__ == "__main__":
    main()

# comand line usage
# python src/tokenizer/train_tokenizer.py --input data/processed/train_formatted.jsonl --vocab_size 20000 --output tokenizer.json