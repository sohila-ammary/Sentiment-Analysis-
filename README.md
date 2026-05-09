# 🎬 IMDb Sentiment Analysis — Traditional NLP

Classify movie reviews as **Positive** or **Negative** using classic NLP techniques.

## What's inside

```
imdb-traditional-nlp/
├── sentiment_analysis.ipynb   # Main notebook (run this)
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"

# 3. Open the notebook
jupyter notebook sentiment_analysis.ipynb
```

## Pipeline

1. **Load** IMDb dataset (25,000 reviews)
2. **Preprocess** — lowercase, remove punctuation, remove stopwords
3. **Vectorize** — TF-IDF
4. **Train** — Logistic Regression
5. **Evaluate** — Accuracy, Precision, Recall, F1, Confusion Matrix
