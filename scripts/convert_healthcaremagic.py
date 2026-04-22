import json
import re
from tqdm import tqdm
import argparse


# -----------------------------
# BASIC CLEANER
# -----------------------------
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


# -----------------------------
# CONVERTER
# -----------------------------
def convert_hcm_kaggle(input_path, output_path):
    print("📥 Loading Kaggle HealthCareMagic JSON...")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} rows")

    with open(output_path, "w", encoding="utf-8") as out:
        for row in tqdm(data, desc="Converting"):
            user = clean_text(row.get("input", "").strip())
            assistant = clean_text(row.get("output", "").strip())

            if len(user) < 5 or len(assistant) < 5:
                continue

            obj = {
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant}
                ]
            }

            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✅ Saved cleaned JSONL to {output_path}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    convert_hcm_kaggle(args.input, args.output)


# Run 
# python scripts/convert_healthcaremagic.py --input data/raw/HealthCareMagic-100k.json --output data/processed/healthcaremagic.jsonl

