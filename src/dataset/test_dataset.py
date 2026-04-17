from tokenizers import Tokenizer
from src.dataset.dataset import ConversationDataset, collate_batch
from torch.utils.data import DataLoader

tok = Tokenizer.from_file("tokenizer.json")

ds = ConversationDataset(
    path="data/processed/cleaned_anon.jsonl",
    tokenizer=tok,
    max_length=256,
    limit=5
)

dl = DataLoader(ds, batch_size=2, collate_fn=lambda x: collate_batch(x, pad_id=0))

for batch, labels in dl:
    print("Batch shape:", batch.shape)
    print(batch)
    break

# Run
# python -m src.dataset.test_dataset
