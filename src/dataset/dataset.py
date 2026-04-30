import json
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100

class ConversationDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=512):
        self.samples = []
        self.tok = tokenizer
        self.max_length = max_length

        eos_id = tokenizer.token_to_id("<eos>")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                msgs = obj["messages"]

                input_ids = []
                labels = []

                for m in msgs:
                    role = m["role"]
                    content = m["content"].strip()

                    if role == "user":
                        text = f"<user> {content} "
                        ids = self.tok.encode(text).ids

                        input_ids.extend(ids)
                        labels.extend([IGNORE_INDEX] * len(ids))

                    elif role == "assistant":
                        text = f"<assistant> {content} <eos> "
                        ids = self.tok.encode(text).ids

                        input_ids.extend(ids)
                        labels.extend(ids)

                # truncate
                input_ids = input_ids[:self.max_length]
                labels = labels[:self.max_length]

                self.samples.append((input_ids, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, labels = self.samples[idx]

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

def collate_batch(batch, pad_id=0):
    IGNORE_INDEX = -100

    max_len = max(len(x[0]) for x in batch)

    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)

    for i, (ids, lbls) in enumerate(batch):
        input_ids[i, : len(ids)] = ids
        labels[i, : len(lbls)] = lbls

    return input_ids, labels
