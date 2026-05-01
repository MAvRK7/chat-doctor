import json
import argparse
from tqdm import tqdm


def format_conversation(messages):
    text = ""

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "").strip()

        if not content:
            continue

        content = " ".join(content.split())

        if role == "user":
            text += "<user> " + content + "\n"
        elif role == "assistant":
            text += "<assistant> " + content + "\n"

    text += "<eos>"
    return text


def process_file(input_path, output_path):
    print(f"\n📥 Loading: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line in tqdm(f_in, desc="Formatting"):
            data = json.loads(line)

            messages = data.get("messages", [])
            if len(messages) < 2:
                continue

            text = format_conversation(messages)

            if len(text) < 20:
                continue

            json.dump({"text": text}, f_out, ensure_ascii=False)
            f_out.write("\n")

    print(f"\n✅ Done writing JSONL: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    process_file(args.input, args.output)

# run
# python scripts/format.py --input data/processed/train.jsonl --output data/processed/train_formatted.jsonl
# python scripts/format.py --input data/processed/val.jsonl --output data/processed/val_formatted.jsonl