import json
import argparse
from collections import Counter
import statistics


def load_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "messages" in obj:
                    data.append(obj)
            except:
                continue
    return data


def analyze(data):
    total_convos = len(data)
    total_messages = 0

    role_counter = Counter()
    msg_lengths = []
    convo_lengths = []
    empty_messages = 0

    for convo in data:
        messages = convo.get("messages", [])
        convo_lengths.append(len(messages))

        for msg in messages:
            total_messages += 1

            role = msg.get("role", "unknown")
            role_counter[role] += 1

            content = msg.get("content", "").strip()

            if not content:
                empty_messages += 1
                continue

            msg_lengths.append(len(content))

    print("\n📊 DATASET OVERVIEW")
    print("=" * 40)
    print(f"Conversations: {total_convos}")
    print(f"Total messages: {total_messages}")
    print(f"Avg messages per convo: {round(statistics.mean(convo_lengths), 2)}")

    print("\n👥 ROLE DISTRIBUTION")
    print("=" * 40)
    for role, count in role_counter.items():
        print(f"{role}: {count}")

    print("\n📝 MESSAGE LENGTHS (characters)")
    print("=" * 40)
    if msg_lengths:
        print(f"Avg: {int(statistics.mean(msg_lengths))}")
        print(f"Median: {int(statistics.median(msg_lengths))}")
        print(f"Max: {max(msg_lengths)}")
        print(f"Min: {min(msg_lengths)}")

    print("\n⚠️ DATA QUALITY")
    print("=" * 40)
    print(f"Empty messages: {empty_messages}")

    # Long message detection
    long_msgs = [l for l in msg_lengths if l > 2000]
    print(f"Messages >2000 chars: {len(long_msgs)}")

    # Short message detection
    short_msgs = [l for l in msg_lengths if l < 20]
    print(f"Messages <20 chars: {len(short_msgs)}")

    multi_turn = sum(1 for c in data if len(c["messages"]) > 2)
    print(f"Multi-turn conversations: {multi_turn}")



def show_samples(data, n=3):
    print("\n🔍 SAMPLE CONVERSATIONS")
    print("=" * 40)

    for i, convo in enumerate(data[:n]):
        print(f"\n--- Conversation {i+1} ---")
        for msg in convo.get("messages", []):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:250]
            print(f"{role.upper()}: {content}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--samples", type=int, default=3)

    args = parser.parse_args()

    print("📥 Loading dataset...")
    data = load_data(args.input)

    print(f"Loaded {len(data)} conversations")

    analyze(data)
    show_samples(data, args.samples)


if __name__ == "__main__":
    main()

# Run 
# python scripts/analyze_dataset.py --input data/processed/train.jsonl --samples 5