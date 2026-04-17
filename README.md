# chat-doctor
A fully local language model built from scratch for medical purposes 

---

📂 File Structure 

```text
chat-doctor/
│
├── data/                     # ignored
│   ├── raw/
│   │   └── train.csv
│   └── processed/
│       └── cleaned_anon.jsonl
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

[Kaggle dataset](https://www.kaggle.com/datasets/thedevastator/medical-conversation-corpus-100k?select=train.csv)

2. Place the file

Move the downloaded file to:

```
data/raw/train.csv
```

3. Process the dataset

Run the preprocessing script:

```
python src/preprocessing/clean_dataset.py --input data/raw/train.csv --output data/processed/cleaned_anon.jsonl
```

⚠️ Note: The data/ directory is ignored in Git due to file size limits, so you must download the dataset locally before running the project.


---