"""
sentiment_analysis.py
---------------------
IMDb Sentiment Analysis — Traditional NLP Pipeline
Steps:
  1. Load dataset
  2. Explore data
  3. Preprocess text
  4. TF-IDF vectorization
  5. Train Logistic Regression
  6. Evaluate (Accuracy, Precision, Recall, F1, Confusion Matrix)
"""

import re
import string
import os

import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# ── Setup ──────────────────────────────────────────────────────────────────────

nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
os.makedirs("results", exist_ok=True)


# ── 1. Load Dataset ────────────────────────────────────────────────────────────

print("\n" + "="*50)
print("STEP 1 — Loading IMDb Dataset")
print("="*50)

train_df = pd.DataFrame(load_dataset("imdb", split="train"))
test_df  = pd.DataFrame(load_dataset("imdb", split="test"))

train_df["sentiment"] = train_df["label"].map({0: "negative", 1: "positive"})
test_df["sentiment"]  = test_df["label"].map({0: "negative", 1: "positive"})

print(f"Train samples : {len(train_df):,}")
print(f"Test  samples : {len(test_df):,}")


# ── 2. Explore Data ────────────────────────────────────────────────────────────

print("\n" + "="*50)
print("STEP 2 — Data Exploration")
print("="*50)

# Class distribution
counts = train_df["sentiment"].value_counts()
print("\nClass distribution:")
print(counts.to_string())

# Average review length
train_df["word_count"] = train_df["text"].str.split().str.len()
print(f"\nAverage review length : {train_df['word_count'].mean():.0f} words")
print(f"Median review length  : {train_df['word_count'].median():.0f} words")

# 5 positive reviews
print("\n--- 5 Positive Reviews ---")
for text in train_df[train_df["label"] == 1]["text"].head(5):
    print(" »", text[:120].replace("\n", " "), "...\n")

# 5 negative reviews
print("--- 5 Negative Reviews ---")
for text in train_df[train_df["label"] == 0]["text"].head(5):
    print(" »", text[:120].replace("\n", " "), "...\n")

# Plot class distribution
counts.plot(kind="bar", color=["#e74c3c", "#2ecc71"], edgecolor="black", figsize=(5, 3))
plt.title("Class Distribution (Train)")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("results/class_distribution.png", dpi=150)
plt.close()
print("Saved → results/class_distribution.png")

# Plot review length
train_df["word_count"].clip(upper=800).hist(bins=40, color="steelblue", edgecolor="white", figsize=(7, 3))
plt.title("Review Length Distribution")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("results/review_lengths.png", dpi=150)
plt.close()
print("Saved → results/review_lengths.png")


# ── 3. Preprocess Text ─────────────────────────────────────────────────────────

print("\n" + "="*50)
print("STEP 3 — Text Preprocessing")
print("="*50)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)                          # remove HTML
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\d+", "", text)                                # remove digits
    tokens = word_tokenize(text)                                   # tokenize
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]  # remove stopwords
    return " ".join(tokens)

# Show before/after example
sample = train_df["text"].iloc[0]
print("\nOriginal  :", sample[:150], "...")
print("Processed :", clean_text(sample)[:150], "...")

print("\nCleaning training reviews...")
X_train = train_df["text"].apply(clean_text)
print("Cleaning test reviews...")
X_test  = test_df["text"].apply(clean_text)

y_train = train_df["label"].values
y_test  = test_df["label"].values
print("Done ✅")


# ── 4. TF-IDF Vectorization ────────────────────────────────────────────────────

print("\n" + "="*50)
print("STEP 4 — TF-IDF Vectorization")
print("="*50)

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),    # unigrams + bigrams
    max_features=50_000,
    min_df=2,
    sublinear_tf=True,     # log scaling
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

print(f"Vocabulary size : {len(tfidf.vocabulary_):,}")
print(f"Train matrix    : {X_train_tfidf.shape}")
print(f"Test  matrix    : {X_test_tfidf.shape}")


# ── 5. Train Model ─────────────────────────────────────────────────────────────

print("\n" + "="*50)
print("STEP 5 — Training Logistic Regression")
print("="*50)

model = LogisticRegression(max_iter=1000, C=1.0)
model.fit(X_train_tfidf, y_train)
print("Training complete ✅")


# ── 6. Evaluate ────────────────────────────────────────────────────────────────

print("\n" + "="*50)
print("STEP 6 — Evaluation")
print("="*50)

y_pred = model.predict(X_test_tfidf)

acc            = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")

print(f"\nAccuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-Score  : {f1:.4f}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — Logistic Regression")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150)
plt.close()
print("Saved → results/confusion_matrix.png")


# ── 7. Try Custom Reviews ──────────────────────────────────────────────────────

print("\n" + "="*50)
print("STEP 7 — Try Custom Reviews")
print("="*50)

def predict(review):
    cleaned = clean_text(review)
    vector  = tfidf.transform([cleaned])
    result  = model.predict(vector)[0]
    prob    = model.predict_proba(vector)[0][result]
    label   = "POSITIVE ✅" if result == 1 else "NEGATIVE ❌"
    print(f"  [{label}  {prob:.2%}]  {review}")

predict("This movie was absolutely amazing, I loved every second of it!")
predict("Terrible film. Boring plot, bad acting. Total waste of time.")
predict("It was okay, nothing special but not bad either.")

print("\n✅ All done! Results saved in ./results/")
