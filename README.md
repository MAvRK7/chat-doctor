# chat-doctor
A fully local language model built from scratch for medical purposes 

🧾 Quick facts:

20M parameter MoE model with SwiGLU. Total samples: 221318. Total tokens: 51,757,583. Average tokens per sample: 233.86. Vocab size: 20,000


Dataset 

A custom dataset has been created for this model. It consists of:

* MedDialogue: 542 (0.23%)
* Medical Conversation Corpus (100k) (MCC): 106378 (45.66%)
* HealthcareMagic: 108690 (46.65%)
* MedQuAD: 16407 (7.04%)
* Greetings, Identity and Refusal (GIR): 950 (0.41%)

    Breakdown of Greetings/Identity/Refusal:
    - Greeting samples: 500 (52.63%)
    - Identity samples: 150 (15.79%)
    - Refusal samples: 300 (31.58%)

Total dataset size: 232967

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
│   │   └── english-train.json (train of MedDialogue)
│   │   └── english-dev.json(val set of MedDialogue)
│   │   └── data/raw/HealthCareMagic-100k.json
│   │   └── data/raw/medquad.csv
│   │
│   └── processed/
│       └── merged.jsonl
│       └── train.jsonl (95% of merged.jsonl)
│       └── val.jsonl (5%)
│       └── healthcaremagic.jsonl
│       └── meddialog_dev.jsonl
│       └── medquad.jsonl
│       └── raw_clean.jsonl
│
├── src/
│   ├──__init__.py
│   ├── tokenizer.py/
│   │   └── count_tokens.py
│   │   └── train_tokenizer.py
│   │   └── verify_tokenizer.py
│   │   └── sample_token_corpus.py
│   │   └── tokenizer.json.model
│   │   └── tokenizer.json.vocab
│   │   └── tokenizer.json_corpus.txt  # ignored
│   │   └── sample_token_corpus.py
│   │   └── tokenizer_sampled.json_corpus.txt
│   │
│   ├── dataset/
│   │   └── dataset.py
│   │   └── test_dataset.py
│   │
│   ├── model/
│   │   └── moe.py
│   │   └── transformer.py
│   │
│   ├── scripts/
│   │   └── convert_csv_to_jsonl.py
│   │   └── convert_healthcaremagic.py
│   │   └── convert_medquad.py
│   │   └── dataset_cleaner.py
│   │   └── gen_multi_geetings.py
│   │   └── merge_datasets.py
│   │   └── analyze_dataset.py
│   │   └── split_cleaned_jsonl.py
│   │
│   ├── sampling.py
│   ├── inference.py
│   ├── train.py
│   ├── agent.py
│   └── utils/
│       ├── config.py
│       └── logging.py
│
├── config.yaml
├── requirements.txt
├── model.pt (the weights) # ignored
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


1. Download the dataset

Download train.csv manually from:

[train dataset](https://www.kaggle.com/datasets/satvikraghav/cleaned-anon-jsonl/data?select=train.jsonl)

[validation dataset](https://www.kaggle.com/datasets/satvikraghav/cleaned-anon-jsonl/data?select=val.jsonl)

2. Place the file

Move the downloaded train file to:

```
data/processed/train.jsonl
```

and place the val dataset in 

```
data/processed/val.jsonl
```

The split in the train dataset (train.jsonl) into train (95%) and validation (val.jsonl) is a 95/5 split 

 - Train size: 221,318 samples
 - Val size: 11,649 samples

⚠️ Note: The data/ directory is ignored in Git due to file size limits, so you must download the dataset locally before running the project.


---