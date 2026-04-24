import json
import os
import re
import random

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(REPO_ROOT, "data/raw/test.jsonl")
OUT_PATH = os.path.join(REPO_ROOT, "data/test/in_domain.jsonl")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

HUMAN_TAG = "[|Human|]"
AI_TAG = "[|AI|]"


def clean_text(t):
    # Remove everything after accidental browser metadata
    t = t.split("edge_all_open_tabs")[0]
    return t.strip()


def extract_pair(text):
    """
    Extract the LAST Human → AI pair.
    """
    text = clean_text(text)

    # Split on Human turns
    human_turns = re.split(re.escape(HUMAN_TAG), text)
    if len(human_turns) < 2:
        return None

    last_human_segment = human_turns[-1]

    # Split that segment by AI tag
    parts = re.split(re.escape(AI_TAG), last_human_segment)
    if len(parts) < 2:
        return None

    human_msg = parts[0].strip()
    ai_msg = parts[1].strip()

    if not human_msg or not ai_msg:
        return None

    return human_msg, ai_msg


def main():
    examples = []

    # Step 1: Collect all valid examples
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("text", "")

            pair = extract_pair(text)
            if pair is None:
                continue

            human, ai = pair

            examples.append({
                "input": f"Patient: {human}\nDoctor:",
                "target": f"Doctor: {ai}"
            })

    # Step 2: Shuffle and sample
    random.seed(42)
    random.shuffle(examples)

    MAX_SAMPLES = 200
    examples = examples[:MAX_SAMPLES]

    # Step 3: Reassign clean IDs
    for i, ex in enumerate(examples):
        ex["id"] = f"in_domain_{i:05d}"

    # Step 4: Write output
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for ex in examples:
            out.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} in-domain examples to {OUT_PATH}")


if __name__ == "__main__":
    main()

# Run
# python -m scripts.build_in_domain_test