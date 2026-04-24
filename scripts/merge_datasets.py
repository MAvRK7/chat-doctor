import json
import argparse
import random
import os
from tqdm import tqdm

def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except:
                continue
    return items


def merge_files(files, output_file):
    all_items = []

    print("\n📥 Loading datasets...")
    for fpath in files:
        print(f" → {fpath}")
        items = load_jsonl(fpath)
        all_items.extend(items)

    print(f"\n🔀 Shuffling {len(all_items)} conversations...")
    random.shuffle(all_items)

    print(f"\n💾 Writing merged dataset → {output_file}")
    with open(output_file, "w", encoding="utf-8") as out:
        for item in tqdm(all_items, desc="Saving"):
            out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ Done. Total merged conversations: {len(all_items)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--files", nargs="+", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    merge_files(args.files, args.out)

# Run 
'''
python scripts/merge_datasets.py \
    --out data/processed/merged.jsonl \
    --files data/processed/meddialog_train.jsonl \
            data/processed/meddialog_dev.jsonl \
            data/processed/raw_clean.jsonl \
            data/processed/healthcaremagic.jsonl \
            data/processed/medquad.jsonl \
            data/processed/combined_greetings_identity.jsonl \
            data/processed/adversarial.jsonl \
            data/processed/mental_health.jsonl 

'''