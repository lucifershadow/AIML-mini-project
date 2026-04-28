# 🎭 Sarcasm Detection in Social Media using Multi-Task Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Description

Sarcasm detection in social media is a challenging natural language processing problem because sarcastic statements often depend on **context** and **implicit sentiment**. Traditional text classification models fail to capture these nuances.

This project develops a **context-aware sarcasm detection model** using **Multi-Task Learning (MTL)**, where a shared BERT encoder jointly learns:

1. **Sarcasm Detection** — binary classification (sarcastic / not sarcastic)
2. **Sentiment Analysis** — 3-class classification (negative / neutral / positive)

The key insight is that sarcasm frequently involves a **contradiction** between positive language and negative context (e.g., *"I absolutely love being stuck in traffic"*). By training the model to also understand sentiment, the sarcasm head gains a richer feature representation.

---

## 📂 Repository Structure

```
sarcasm-detection-mtl/
│
├── sarcasm_detection_mtl.py         # Full read-only source code
├── sarcasm_detection_complete.ipynb # Google Colab notebook (runnable)
├── README.md                        # This file
│
└── screenshots/
    ├── 01_eda_class_distribution.png
    ├── 02_eda_word_count.png
    ├── 03_sentiment_distribution.png
    ├── 04_bert_confusion_matrix.png
    ├── 05_mtl_confusion_matrix.png
    ├── 06_model_comparison_bar.png
    ├── 07_full_metric_comparison.png
    ├── 08_all_confusion_matrices.png
    └── 09_inference_demo_output.png
```

---

## 🧩 Modules Implemented

### Module 1 — Data Collection & Loading
- **Sarcasm Headlines Dataset v2** (28,000+ news headlines, manually labelled)
- **Conversational Sarcasm Dataset** (`maliha/sarcasm-explain-5k` from HuggingFace)
- Fallback to Twitter sarcasm dataset (`shiv213/Automatic-Sarcasm-Detection-Twitter`)
- Combined dataset of ~7,000+ balanced samples after sampling
- **Data Augmentation**: 27 hand-crafted sarcastic + 12 non-sarcastic sentences ×8 repetitions

### Module 2 — Exploratory Data Analysis (EDA)
- Class distribution bar chart
- Word count distribution by class (histogram + box plot)
- Top-15 most frequent words per class (stopword-filtered)
- Sentiment distribution per sarcasm class (bar chart)

### Module 3 — Text Preprocessing
- Lowercasing
- URL removal (`http\S+`)
- Special character / punctuation removal
- Whitespace normalization
- BERT tokenization with padding and truncation (`max_length=64`)

### Module 4 — Stratified Train / Val / Test Split
- 70% Training / 15% Validation / 15% Test
- Stratified split to maintain class balance across all splits
- Class weight computation for imbalanced learning

### Module 5 — Baseline 1: TF-IDF + Logistic Regression
- `TfidfVectorizer` with 10,000 features
- `LogisticRegression` with balanced class weights
- Evaluated with all 4 metrics

### Module 6 — Baseline 2: BERT Single-Task
- `bert-base-uncased` fine-tuned with HuggingFace `Trainer` API
- 4 training epochs, batch size 16, warmup + weight decay
- Best model checkpoint loaded via `load_best_model_at_end=True`

### Module 7 — Sentiment Pipeline (RoBERTa)
- Pre-trained `cardiffnlp/twitter-roberta-base-sentiment-latest` used as frozen pipeline
- Generates 3-dimensional probability vectors [neg, neu, pos] for every sample
- Sentiment probabilities cached upfront to avoid re-computation during training

### Module 8 — Multi-Task Learning Model (Proposed)
- **Shared BERT encoder** (`bert-base-uncased`) — top 4 layers + pooler fine-tuned
- **Sarcasm head**: Linear(771→256) → GELU → Dropout → Linear(256→2)
  - Input: BERT [CLS] (768-d) + sentiment probs (3-d) = 771-d
- **Sentiment head**: Linear(768→256) → GELU → Dropout → Linear(256→3)
- Combined loss: `L = α × L_sarcasm + (1−α) × L_sentiment` (α = 0.70)
- Label smoothing (0.1) + gradient clipping + mixed precision (fp16)

### Module 9 — Inference & Ensemble
- Rule-based lexicon detector (`SARCASM_LEXICON` + positive/negative word matching)
- Ensemble score: `max(bert_prob, rule_score)`
- Optimal threshold search on validation set (range 0.30–0.65)

### Module 10 — Evaluation & Comparison
- All 3 models evaluated on: **Accuracy, Precision, Recall, F1-Score**
- Confusion matrices for all 3 models (side-by-side)
- Bar charts comparing all 4 metrics across models
- Qualitative analysis of MTL improvement

---

## 🛠️ Technologies Used

| Category | Technology |
|---|---|
| **Language** | Python 3.9+ |
| **Deep Learning** | PyTorch 2.x |
| **Transformers** | HuggingFace Transformers 4.x |
| **Pre-trained Models** | `bert-base-uncased`, `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| **Datasets** | HuggingFace `datasets` library |
| **ML Utilities** | scikit-learn (TF-IDF, Logistic Regression, metrics, class weights) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Training Runtime** | Google Colab (GPU — T4 / A100) |
| **Mixed Precision** | `torch.cuda.amp.GradScaler` |
| **Optimization** | AdamW + Linear LR Scheduler with Warmup |

---

## 🏗️ Model Architecture

```
                    Input Text
                        │
               [BERT Tokenizer]
                        │
            ┌───────────────────────┐
            │   Shared BERT Encoder  │
            │  (bert-base-uncased)   │
            │  Top 4 layers + pooler │
            │   fine-tuned           │
            └───────────┬───────────┘
                        │
                  [CLS] pooled
                  output (768-d)
                   ┌────┴────┐
                   │         │
          [Task 0]           [Task 1]
         Sarcasm             Sentiment
           Head               Head
        (768 + 3               (768-d)
       = 771-d) ←──────────────────────
          │        Sentiment         │
     Binary out    Prob Vector    3-class out
    (Sarc/Not)     [neg,neu,pos]  (Neg/Neu/Pos)
```

---

## 📊 Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Baseline 1: TF-IDF + Logistic Regression | — | — | — | — |
| Baseline 2: BERT (single-task) | — | — | — | — |
| **Proposed: BERT MTL + Sentiment** | — | — | — | — |

> *Run the notebook to fill in actual scores from your Colab session.*

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended)
1. Upload `sarcasm_detection_complete.ipynb` to Google Colab
2. Upload `Sarcasm_Headlines_Dataset_v2.json` to your Google Drive
3. Enable GPU: `Runtime → Change runtime type → T4 GPU`
4. Run all cells sequentially

### Option 2: Local (requires GPU)
```bash
pip install torch transformers datasets scikit-learn seaborn matplotlib pandas
python sarcasm_detection_mtl.py
```

---

## 📷 Output Screenshots

See the [`screenshots/`](./screenshots/) folder for:
- EDA plots (class distribution, word counts)
- Sentiment distribution per class
- Confusion matrices (all 3 models)
- 4-metric comparison bar chart
- Inference demo output

---

## 📝 Key Findings

1. **TF-IDF + LR** is fast and interpretable but misses contextual sarcasm cues entirely.
2. **BERT (single-task)** significantly outperforms the baseline by capturing contextual semantics.
3. **BERT MTL** leverages the positive-words-in-negative-context pattern via explicit sentiment features.
4. The **rule-based ensemble** further boosts recall on canonical sarcasm patterns with negligible extra compute.

---

## 📚 References

- Devlin et al. (2019) — *BERT: Pre-training of Deep Bidirectional Transformers*
- Misra & Arora (2023) — *Sarcasm Detection using News Headlines Dataset*
- Barbieri et al. (2020) — *TweetEval: Unified Benchmark for Tweet Classification*
- HuggingFace Transformers Documentation — https://huggingface.co/docs/transformers
