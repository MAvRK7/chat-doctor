# chat-doctor
A fully local language model built from scratch for medical purposes 

🧾 Quick facts:

20M parameter MoE model with SwiGLU, 27 million‑token dataset.

---

📂 File Structure 

```text
chat-doctor/
│
├── data/                     # ignored
│   ├── raw/
│   │   └── train.csv
│   │   └── test.csv
│   │   └── test.jsonl (for final test)
│   └── processed/
│       └── cleaned_anon.jsonl
│       └── train.jsonl (95% of cleaned_anon.jsonl)
│       └── val.jsonl (5%)
│
├── src/
│   ├──__init__.py
│   ├── preprocessing/
│   │   └── clean_dataset.py
│   │   └── train_tokenizer.py   
│   ├── tokenizer.py/
│   │   └── count_tokens.py
│   │   └── train_tokenizer.py
│   │   └── verify_tokenizer.py
│   ├── dataset/
│   │   └── dataset.py
│   │   └── test_dataset.py
│   ├── model/
│   │   └── moe.py
│   │   └── transformer.py
│   ├── scripts/
│   │   └── convert_csv_to_jsonl.py
│   │   └── split_cleaned_jsonl.py
│   ├── train.py
│   ├── agent.py
│   └── utils/
│       ├── config.py
│       └── logging.py
├── config.yaml
├── requirements.txt
├── tokenizer.json
└── README.md

```
---

## 🛠️ Installation

### 1️⃣ Clone the repository

```git 
git clone https://github.com/MAvRK7/chat-doctor.git
cd chat-doctor
```
### 2️⃣ Create and activate a virtual environment:

```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```
### 4️⃣ 📦 Dataset Setup

This project uses the Medical Conversation Corpus (100k) from Kaggle.

1. Download the dataset

Download train.csv manually from:

[Raw train dataset](https://www.kaggle.com/datasets/thedevastator/medical-conversation-corpus-100k?select=train.csv)

or Directly download the pre-processed dataset - cleaned_anon.jsonl

[Processed train dataset](https://www.kaggle.com/datasets/satvikraghav/cleaned-anon-jsonl/data)

Similarly download the test dataset from

[Raw test dataset](https://www.kaggle.com/datasets/thedevastator/medical-conversation-corpus-100k?select=test.csv)

or directly download the test dataset in jsonl format

[Processed test dataset](https://www.kaggle.com/datasets/satvikraghav/cleaned-anon-jsonl/data/data/settings/settings/settings/settings/settings/settings/settings/settings/settings/settings/settings/settings?select=test.jsonl)

2. Place the file

Move the downloaded Raw train file to:

```
data/raw/train.csv
```

OR place the processed dataset in 

```
data/processed/cleaned_anon.jsonl
```

Similarly place the raw test file in 

```
data/raw/test.csv
```

OR place the processed dataset in 

```
data/raw/test.jsonl
```

3. Process the dataset (skip if you've downloaded the processed dataset)

Run the preprocessing script:

For train.csv

```
python src/preprocessing/clean_dataset.py --input data/raw/train.csv --output data/processed/cleaned_anon.jsonl
```

For test.csv

```
python scripts/convert_csv_to_jsonl.py --input data/raw/train.csv --output data/raw/test.jsonl
```

4. Then run 
```
python scripts/split_cleaned_jsonl.py
``` 
This will split the train dataset (cleaned_anon.jsonl) into train (95%) and validation (5%) splits

 - Train size: 101,216 samples
 - Val size: 5,328 samples

⚠️ Note: The data/ directory is ignored in Git due to file size limits, so you must download the dataset locally before running the project.


---