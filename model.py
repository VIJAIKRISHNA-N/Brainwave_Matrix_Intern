import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def clean_text(text):
    """Clean text by removing punctuation, links, and formatting."""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def load_and_prepare_data():
    """Load fake and real news data and prepare it for training."""
    df_fake = pd.read_csv("Fake.csv")
    df_real = pd.read_csv("True.csv")

    df_fake["label"] = 0
    df_real["label"] = 1

    df = pd.concat([df_fake, df_real], axis=0)
    df = df[["text", "label"]]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df["text"] = df["text"].apply(clean_text)

    return df

def train_model(X_train, y_train):
    """Train a logistic regression model on the TF-IDF features."""
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print performance metrics."""
    y_pred = model.predict(X_test)
    print("✅ Model Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

def predict_news(text, mode_
