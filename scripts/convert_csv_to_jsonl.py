import csv
import json
import os
import argparse


def convert(input_file, output_file):
    with open(input_file, "r", encoding="utf-8", newline="") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)

        for row in reader:
            text = row.get("Conversation")
            if not text:
                continue

            # 🔥 KEEP EXACTLY AS IS
            json_obj = {"text": text}
            f_out.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

    print(f"✅ Done. Saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    convert(args.input, args.output)

# Run
# for test 
# python scripts/convert_csv_to_jsonl.py --input data/raw/train.csv --output data/raw/test.jsonl
