import json
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


class ConversationDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=512):
        self.samples = []
        self.tok = tokenizer
        self.max_length = max_length

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                text = obj.get("text", "")

                if "<user>" not in text or "<assistant>" not in text:
                    continue

                user_part = text.split("<user>")[1].split("<assistant>")[0].strip()
                assistant_part = text.split("<assistant>")[1].split("<eos>")[0].strip()

                input_ids = []
                labels = []

                # USER (no loss)
                user_tokens = self.tok.encode(f"<user> {user_part} ")

                input_ids.extend(user_tokens)
                labels.extend([IGNORE_INDEX] * len(user_tokens))

                # ASSISTANT PROMPT (no loss)
                assistant_prompt = self.tok.encode("<assistant> ")

                input_ids.extend(assistant_prompt)
                labels.extend([IGNORE_INDEX] * len(assistant_prompt))

                # ASSISTANT ANSWER (TRAIN HERE ONLY)
                answer_tokens = self.tok.encode(assistant_part)

                input_ids.extend(answer_tokens)
                labels.extend(answer_tokens)

                # truncate
                input_ids = input_ids[:self.max_length]
                labels = labels[:self.max_length]

                self.samples.append((input_ids, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)


def collate_batch(batch, pad_id=0):
    max_len = max(len(x[0]) for x in batch)

    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)

    for i, (ids, lbls) in enumerate(batch):
        input_ids[i, :len(ids)] = ids
        labels[i, :len(lbls)] = lbls

    return input_ids, labels