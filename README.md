# Emotion Classification NLP

A simple machine learning project that classifies text into emotions (e.g. joy, sadness, anger, fear, love, surprise) using classic NLP techniques.

## Overview

This project takes short pieces of text and predicts the emotion behind them. It walks through the full pipeline: cleaning raw text, converting it into numerical features, and training models to classify it.

## What It Does

1. **Data Loading** — Reads labeled text data (`text` + `emotion`)
2. **Preprocessing**
   - Lowercasing
   - Removing punctuation
   - Removing numbers
   - Removing non-ASCII characters (emojis, etc.)
   - Removing stopwords
3. **Feature Extraction** — Converts text into numbers using:
   - Bag of Words (`CountVectorizer`)
   - TF-IDF (`TfidfVectorizer`)
4. **Modeling** — Trains and compares:
   - Multinomial Naive Bayes (on BoW)
   - Multinomial Naive Bayes (on TF-IDF)
   - Logistic Regression (on BoW)
5. **Evaluation** — Measures accuracy on a held-out test set

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
nltk
```

Install with:
```bash
pip install -r requirements.txt
```

You'll also need to download NLTK's stopwords and tokenizer data (the notebook does this automatically):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## How to Run

1. Clone the repo
2. Make sure `train.txt` is in the same folder as the notebook
3. Open `emotion.ipynb` in Jupyter Notebook or Google Colab
4. Run all cells

## ataset

The dataset (`train.txt`) contains short text samples paired with an emotion label, separated by `;`.

## Notes

This project was built for practice — to get hands-on experience with text preprocessing, feature extraction, and comparing simple classification models for NLP tasks.
