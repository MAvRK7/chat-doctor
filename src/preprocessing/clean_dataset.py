import re
import json
import argparse

'''It deletes all double quotes from the text, removes a specific newline+quote pattern, 
and trims whitespace from the start and end.'''

def clean_conversation(conversation_text):
    """Basic cleanup."""
    conversation_text = conversation_text.replace('\n"', "").replace('"', "")
    return conversation_text.strip()

# --- ANONYMISATION HELPERS ---

# Let this remain 
WHITELIST = {
    "I", "My", "The", "A", "An", "And", "But", "Or",
    "Hi", "Hello", "Thanks", "Thank",
    "Doctor", "Dr", "MRI", "EEG", "ICU",
    "Feb", "February", "Mar", "March", "Apr", "April",
    "May", "Jun", "June", "Jul", "July", "Aug", "August",
    "Sep", "Sept", "September", "Oct", "October",
    "Nov", "November", "Dec", "December",
    "Person", "Patient"
}

LOCATION_WORDS = [
    "Tehran", "Iran", "USA", "UK", "Canada", "Australia", "India",
    "London", "Melbourne", "Sydney", "New York", "Paris", "Berlin"
    # add more as needed
]

def anonymize_text(text: str) -> str:
    # Remove emails
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[EMAIL]', text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)

    # Remove phone numbers
    text = re.sub(r'\b\+?\d[\d\-\s]{6,}\d\b', '[PHONE]', text)

    # Replace known locations
    for loc in LOCATION_WORDS:
        text = re.sub(r'\b' + re.escape(loc) + r'\b', '[LOCATION]', text, flags=re.IGNORECASE)

    # Replace capitalised words that look like names
    def replace_name_like(match):
        word = match.group(0)
        if word in WHITELIST:
            return word
        # If it's at the start of a sentence, keep it
        if match.start() == 0:
            return word
        return "Person"

    # Replace capitalised words NOT in whitelist
    text = re.sub(r'\b([A-Z][a-z]{2,})\b', replace_name_like, text)

    return text

def extract_conversations(text):
    chunks = text.split("The conversation between human and AI assistant.")
    conversations = []

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        # Use the original, more flexible regex pattern
        turns = re.findall(
            r"\[\|Human\|\](.*?)\[\|AI\|\](.*?)(?=\[\|Human\|\]|$)",
            chunk,
            re.DOTALL
        )

        for human, ai in turns:
            human = clean_conversation(human) #first clean, then anonymise
            ai = clean_conversation(ai)

            # anonymise both sides
            human = anonymize_text(human)
            ai = anonymize_text(ai)

            if len(human) < 10 or len(ai) < 10:
                continue

            # JSONL format
            conversations.append({
                "messages": [
                    {"role": "user", "content": human},
                    {"role": "assistant", "content": ai}
                ]
            })

    return conversations

def clean_dataset(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    conversations = extract_conversations(text)

    with open(output_file, "w", encoding="utf-8") as out:
        for conv in conversations:
            out.write(json.dumps(conv, ensure_ascii=False) + "\n")

    print(f"✅ Extracted {len(conversations)} anonymised conversations")

# Run
# chat-doctor/src/preprocessing/clean_dataset.py
# python src/preprocessing/clean_dataset.py --input data/raw/train.csv --output data/processed/cleaned_anon.jsonl

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    clean_dataset(args.input, args.output)
