import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Download stopwords if not present
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("train.csv")
print("Dataset loaded successfully.")

# Fill nulls with empty strings
df.fillna('', inplace=True)

# Combine title, author and text
df['content'] = df['author'] + ' ' + df['title'] + ' ' + df['text']

# Initialize stemmer
ps = PorterStemmer()

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    tokens = text.split()
    tokens = [ps.stem(word) for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

# Apply preprocessing
df['content'] = df['content'].apply(preprocess)

# Feature extraction
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['content']).toarray()
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Optional: test on custom input
def predict_fake_news(input_text):
    input_text = preprocess(input_text)
    vector = tfidf.transform([input_text]).toarray()
    prediction = model.predict(vector)[0]
    return "Fake News" if prediction == 1 else "Real News"

# Example usage
sample = "Breaking: Government confirms discovery of alien spaceship."
print(f"Sample prediction: {predict_fake_news(sample)}")
