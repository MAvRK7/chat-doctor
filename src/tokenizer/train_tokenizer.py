import sentencepiece as spm
import json
import argparse
import random
import re
import os
from tqdm import tqdm  # Importing tqdm for progress bar

def normalize_medical_text(text: str) -> str:
    text = text.strip()

    # separate units (500mg → 500 mg)
    text = re.sub(
        r'(\d+)(mg|ml|kg|g|mcg|mmhg|bpm|cm|mm)',
        r'\1 \2',
        text,
        flags=re.IGNORECASE
    )

    # normalize vitals (120/80 → 120 / 80)
    text = re.sub(r'(\d+)/(\d+)', r'\1 / \2', text)

    return text


def load_jsonl_conversations(path):
    conversations = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            convo = []
            for msg in obj.get("messages", []):
                role = msg.get("role", "").strip()
                content = msg.get("content", "").strip()

                if not content:
                    continue

                content = normalize_medical_text(content)
                convo.append(f"<{role}> {content}")

            if convo:
                conversations.append(" ".join(convo))

    return conversations


def write_corpus(texts, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--sample_size", type=int, default=None)
    args = parser.parse_args()

    all_texts = []

    print("📥 Loading input files...")
    for path in args.input:
        if path.endswith(".jsonl"):
            texts = load_jsonl_conversations(path)
            print(f"Loaded {len(texts)} conversations from {path}")
            all_texts.extend(texts)

    print(f"Total before dedup: {len(all_texts)}")

    # deduplicate
    all_texts = list(set(all_texts))
    print(f"After dedup: {len(all_texts)}")

    # prioritize longer (richer medical content)
    all_texts.sort(key=len, reverse=True)

    # mixed sampling
    if args.sample_size and len(all_texts) > args.sample_size:
        half = args.sample_size // 2
        long_part = all_texts[:half]
        rest = all_texts[half:]
        random.shuffle(rest)
        all_texts = long_part + rest[:half]
        print(f"Subsampled to {len(all_texts)}")
    else:
        random.shuffle(all_texts)

    # Create output directory for tokenizer if it doesn't exist
    tokenizer_dir = "src/tokenizer"
    os.makedirs(tokenizer_dir, exist_ok=True)

    # Write corpus
    corpus_path = os.path.join(tokenizer_dir, "corpus.txt") 
    print("💾 Writing corpus to disk...")
    write_corpus(all_texts, corpus_path)

    # Initialize progress bar for the corpus size
    print("🧠 Training tokenizer with progress bar...")
    
    # Progress bar for tokenization process (before training starts)
    with tqdm(total=len(all_texts), desc="Preparing corpus", unit="sentence") as pbar:
        pbar.update(len(all_texts))  # Simulating progress for corpus preparation

    # Train the SentencePiece tokenizer
    spm.SentencePieceTrainer.Train(
        input=corpus_path,
        model_prefix=os.path.join(tokenizer_dir, args.output),  # Save the model and vocab to src/tokenizer/
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=0.9997,
        input_sentence_size=5000000,
        shuffle_input_sentence=True,
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
            "<system>",
            "<mask>"
        ]
    )

    print(f"✅ Saved tokenizer: {os.path.join(tokenizer_dir, args.output)}.model / {os.path.join(tokenizer_dir, args.output)}.vocab")



if __name__ == "__main__":
    main()

# comand line usage
# python src/tokenizer/train_tokenizer.py --input data/processed/train.jsonl --vocab_size 20000 --output tokenizer.json