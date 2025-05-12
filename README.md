# 🗾 Kotoba Tokenization Demo

**Kotoba** is an interactive Streamlit app for comparing Japanese tokenizers.  
It visualizes token segmentation, shows glosses/translations, and evaluates each tokenizer on curated examples.

---

## 🚀 Features

- **Three tokenizers included**:
  - ✅ SudachiPy (SplitMode.C)
  - ✅ Janome
  - ✅ Custom tokenizer (Sudachi + BERT + rule-based preprocessing)

- **Segmentation visualization**:
  - Tokens are color-coded by character type (Kanji / Hiragana / Katakana / Latin)
  - Glossed with English translations (or romaji fallback)

- **Evaluation with 4 key metrics**:
  1. **Tokenization F1** — overlap with gold tokens
  2. **Morpheme Recall** — how many important morphemes are preserved as complete tokens
  3. **Entity Recall** — how well named entities are preserved as intact tokens
  4. **Back-translation Accuracy** — overlap between original English and reverse translation

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourname/kotoba-tokenization-demo.git
cd kotoba-tokenization-demo
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Initialize Sudachi dictionary (only needed once):

```bash
python -m sudachipy download
```

---

## ▶️ Run the app

```bash
streamlit run main.py
```

---

## 📁 Project Structure

* `main.py` — main Streamlit app
* `ml_curated_examples.json` — curated bilingual examples with styles, entities, and morphemes
* `requirements.txt` — dependencies
* *(optional)* `packages.txt` and `config.toml` for Streamlit Cloud deployment

---

## 📊 Example Output

| Tokenizer | F1 Score | Morpheme Rec. | Entity Rec. | Back-translation |
| --------- | -------- | ------------- | ----------- | ---------------- |
| Sudachi   | 0.78     | 100%          | 100%        | 88.2%            |
| Janome    | 0.71     | 83%           | 100%        | 88.2%            |
| Custom    | 0.85     | 100%          | 100%        | 88.2%            |

---

## 📌 Why?

This app was built as part of a master's thesis in Japanese NLP,
demonstrating that **tokenization** remains the central challenge
for robust language understanding in Japanese.

---
