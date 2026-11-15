# Scientific Paper Classification Using Multi-Model NLP Pipelines

Classifying arXiv papers into six closely related research categories using only their abstracts. The project compares classical machine learning (TF-IDF), recurrent neural networks (Word2Vec), and a fine-tuned Transformer (BERT) under one shared preprocessing, splitting, and evaluation protocol.

**Best result:** BERT-base-uncased reaches **76.02% accuracy / 76.08% macro-F1** on the held-out test set. GRU (76.00%) and Logistic Regression (75.39%) come very close.

## Task

Given a paper's abstract, predict which of these six arXiv categories it belongs to:

| Category | Papers (before capping) |
|---|---|
| Machine Learning | 80,413 |
| Computer Vision and Pattern Recognition | 71,081 |
| Computation and Language (NLP) | 37,243 |
| Artificial Intelligence | 28,679 |
| Machine Learning (Statistics) | 13,886 |
| Neural and Evolutionary Computing | 6,710 |

These categories overlap heavily in vocabulary, which is what makes the task hard (especially *Machine Learning* vs. its neighbours).

## Dataset

- **Source:** arXiv scientific paper dataset (`arXiv_scientific_dataset.csv`), 287,421 papers × 10 columns, no missing values.
- Place the CSV in the project root before running the notebook (it is not bundled in this repo).
- After filtering to the six categories, the data is capped at **12,000 papers per class** to reduce imbalance and training time.
- **Split:** stratified 70 / 15 / 15 → 46,697 train / 10,006 validation / 10,007 test. The validation set is used for hyperparameter selection; the test set is touched once for final evaluation.

## Pipeline

1. **Preprocessing & EDA**: class distribution, abstract-length analysis, word clouds, duplicate-abstract removal.
2. **Text cleaning**: lowercasing, LaTeX/math stripping, stop-word removal, lemmatization (compared against keeping stop-words and stemming on a 2,000-abstract sample). Very short abstracts (<10 tokens after cleaning) are dropped.
3. **Imbalance handling**: per-class cap of 12,000 plus balanced class weights during training.
4. **Models** (3 hyperparameter configurations each, best chosen by validation macro-F1):
   - **Classical (TF-IDF, 20k features, 1–2-grams):** Random Forest, Logistic Regression, Multinomial Naive Bayes
   - **RNN family (self-trained Word2Vec skip-gram, 200-d, max length 256):** SimpleRNN, GRU, LSTM and their bidirectional versions
   - **Transformer:** `bert-base-uncased` fine-tuned (max length 256)
5. **Evaluation**: accuracy, macro-F1, weighted-F1, per-class reports and confusion matrices on the test set.

## Results (test set)

| Rank | Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|
| 1 | **BERT-base** | **76.02%** | **76.08%** | **75.66%** |
| 2 | GRU | 76.00% | 75.99% | 75.58% |
| 3 | Logistic Regression | 75.39% | 75.63% | 75.22% |
| 4 | BiGRU | 75.86% | 75.25% | 74.85% |
| 5 | LSTM | 75.14% | 75.12% | 74.67% |
| 6 | BiLSTM | 75.29% | 74.72% | 74.42% |
| 7 | Naive Bayes | 74.12% | 73.98% | 73.59% |
| 8 | Random Forest | 72.38% | 71.91% | 71.24% |
| 9 | BiSimpleRNN | 71.82% | 71.40% | 71.01% |
| 10 | SimpleRNN | 57.69% | 55.27% | 54.98% |

**Key takeaways**

- BERT is best, but only marginally ahead of GRU and a well-tuned TF-IDF + Logistic Regression baseline.
- Gated RNNs (GRU/LSTM) strongly outperform vanilla SimpleRNN; bidirectionality gave no consistent gain.
- *Machine Learning* is the hardest class (F1 ≈ 0.5–0.6 for most models) because it overlaps with AI, Statistics-ML, and Neural Computing; *Computer Vision* is the easiest (F1 ≈ 0.88).

## Repository Structure

```
.
├── project_code.ipynb        # Full pipeline, organised in five parts:
│                             #   01 Preprocessing & split
│                             #   02 Classical ML (TF-IDF)
│                             #   03 RNN family (Word2Vec)
│                             #   04 BERT fine-tuning
│                             #   05 Final model comparison
├── project_report.pdf        # Full written report
└── arXiv_scientific_dataset.csv   # (add yourself, not included)
```

Running the notebook also generates intermediate files: `arxiv_cleaned_full.csv`, `arxiv_cleaned_capped.csv`, `train_df.csv`, `val_df.csv`, `test_df.csv`, `label_encoder.joblib`, `tuning_log.csv`, and `final_results.csv`. Results are logged to CSV after every run, so the notebook can resume after a kernel restart, and trained models are reloaded instead of retrained.

## Getting Started

### Requirements

- Python 3.9+
- A CUDA-capable GPU is strongly recommended for the RNN and BERT sections

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud joblib \
            gensim tensorflow torch transformers datasets jupyter
```

NLTK resources (stopwords, punkt, wordnet) are downloaded automatically by the notebook.

### Run

```bash
git clone <your-repo-url>
cd <your-repo-name>
# add arXiv_scientific_dataset.csv to this folder
jupyter notebook project_code.ipynb
```

Run the parts in order (01 → 05); later parts reload the files produced by part 01. All experiments use `RANDOM_STATE = 42`.

## Future Work

- Domain-specific Transformers (e.g. SciBERT) with longer sequence lengths
- Using titles and other metadata alongside abstracts
- Multi-label classification, since arXiv papers often belong to several categories

## Authors

- Tasnim Rahmatullah, Dept. of CSE, Brac University
- Zafra Musarrat Islam, Dept. of CS, Brac University
- Md Zubayer Hasan, Dept. of CSE, Brac University
- Karib Fiyaz, Dept. of CS, Brac University

## License

Add a license of your choice (e.g. MIT) before publishing.
