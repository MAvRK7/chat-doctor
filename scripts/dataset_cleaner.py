import re
import json
import argparse
from tqdm import tqdm

# -----------------------------------
# BASIC CLEANING (FAST, NO NER)
# -----------------------------------

def clean_text(text: str) -> str:
    if not text:
        return ""

    # Remove emails
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[EMAIL]', text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)

    # Remove phone numbers
    text = re.sub(r'\b\+?\d[\d\-\s]{6,}\d\b', '[PHONE]', text)

    # Remove common signatures
    text = re.sub(r"thank you.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"regards.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"chatdoctor.*$", "", text, flags=re.IGNORECASE)

    # Fix spacing
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# -----------------------------------
# PROCESS RAW DATASET
# -----------------------------------

def process_raw_dataset(text):
    conversations = []

    blocks = text.split("The conversation between human and AI assistant.")

    for block in tqdm(blocks, desc="Parsing raw dataset"):
        block = block.strip()
        if not block:
            continue

        parts = block.split("[|Human|]")

        for part in parts:
            if "[|AI|]" not in part:
                continue

            human, ai = part.split("[|AI|]", 1)

            human = clean_text(human.strip())
            ai = clean_text(ai.strip())

            if len(human) < 5 or len(ai) < 5:
                continue

            conversations.append({
                "messages": [
                    {"role": "user", "content": human},
                    {"role": "assistant", "content": ai}
                ]
            })

    return conversations


# -----------------------------------
# PROCESS MEDDIALOG
# -----------------------------------

def process_med_dialogue(data):
    conversations = []

    for item in tqdm(data, desc="Processing MedDialog"):
        utts = item.get("utterances", [])
        if len(utts) < 2:
            continue

        user = clean_text(utts[0].replace("patient:", "").strip())
        assistant = clean_text(utts[1].replace("doctor:", "").strip())

        if len(user) < 5 or len(assistant) < 5:
            continue

        conversations.append({
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant}
            ]
        })

    return conversations


# -----------------------------------
# MAIN PIPELINE
# -----------------------------------

def clean_dataset(input_file, output_file):
    print(f"\n📥 Loading: {input_file}")

    if input_file.endswith(".json") or input_file.endswith(".jsonl"):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        conversations = process_med_dialogue(data)

    else:
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()
        conversations = process_raw_dataset(text)

    print(f"\n💾 Writing: {output_file}")

    with open(output_file, "w", encoding="utf-8") as f:
        for conv in tqdm(conversations, desc="Saving JSONL"):
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    print(f"\n✅ Done. Total conversations: {len(conversations)}")
    

# -----------------------------------
# CLI
# -----------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    clean_dataset(args.input, args.output)


# Run 
# Dataset 1 (MedDialogue)
# Train
# python scripts/dataset_cleaner.py --input data/raw/english-train.json --output data/processed/meddialog_train.jsonl
# Val
# python scripts/dataset_cleaner.py --input data/raw/english-dev.json --output data/processed/meddialog_dev.jsonl
#---
# Dataset 2 (Medical Conversation Corpus (100k+))
# python scripts/dataset_cleaner.py --input data/raw/train.csv --output data/processed/raw_clean.jsonl
