import pandas as pd
import json
import re
from tqdm import tqdm
import argparse


def clean_text(text):
    if not text:
        return ""

    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[EMAIL]', text)
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    text = re.sub(r'\b\+?\d[\d\-\s]{6,}\d\b', '[PHONE]', text)
    text = re.sub(r"thank you.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"regards.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def convert_medquad(input_path, output_path):
    print("📥 Loading MedQuAD CSV...")
    df = pd.read_csv(input_path)

    # Confirm columns
    if not {"question", "answer"}.issubset(df.columns):
        raise ValueError(f"❌ CSV missing required columns. Found: {df.columns}")

    print(f"Loaded {len(df)} rows")

    with open(output_path, "w", encoding="utf-8") as out:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting"):
            q = clean_text(str(row["question"]).strip())
            a = clean_text(str(row["answer"]).strip())

            if len(q) < 5 or len(a) < 5:
                continue

            obj = {
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ]
            }

            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✅ Saved cleaned MedQuAD JSONL to {output_path}")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    convert_medquad(args.input, args.output)

# Run 
'''
python scripts/convert_medquad.py \
    --input data/raw/medquad.csv \
    --output data/processed/medquad.jsonl
'''