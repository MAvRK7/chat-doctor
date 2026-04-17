import json
import torch
from torch.utils.data import Dataset
class ConversationDataset(Dataset):
    """
    Loads your JSONL dataset and converts each conversation into a single
    training string: "User: ... Assistant: ...".
    Tokenization happens inside __getitem__.
    """

    def __init__(self, path, tokenizer, max_length=256, limit=None):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break

                obj = json.loads(line)
                msgs = obj["messages"]

                # Build a single training string
                text = ""
                for m in msgs:
                    prefix = "User: " if m["role"] == "user" else "Assistant: "
                    text += prefix + m["content"].strip() + "\n"

                self.samples.append(text)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]

        enc = self.tokenizer.encode(text)
        ids = enc.ids[: self.max_length]

        # Convert to tensor
        ids = torch.tensor(ids, dtype=torch.long)

        return ids


def collate_batch(batch, pad_id=0):
    """
    Pads a batch of variable-length sequences into a rectangular tensor.
    """
    max_len = max(len(x) for x in batch)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)

    for i, seq in enumerate(batch):
        padded[i, : len(seq)] = seq

    # Labels are the same as inputs (next-token prediction)
    return padded, padded.clone()
