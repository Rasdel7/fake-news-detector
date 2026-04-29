import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))


print("Loading datasets...")
fake = pd.read_csv('Fake.csv')
real = pd.read_csv('True.csv')

fake['label'] = 0
real['label'] = 1

df = pd.concat([fake, real], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total articles: {len(df):,}")
print(f"Fake: {len(fake):,} | Real: {len(real):,}")


df['content'] = df['title'] + ' ' + df['text']
df['content'] = df['content'].str.lower()


X = df['content']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


print("\nVectorizing text...")
tfidf = TfidfVectorizer(
    max_features=50000,
    stop_words='english',
    ngram_range=(1, 2)
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)


print("Training model...")
model = LogisticRegression(max_iter=1000, C=1.0)
model.fit(X_train_tfidf, y_train)


preds    = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, preds)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, preds,
      target_names=['Fake', 'Real']))

print("\nSaving model...")
with open('model.pkl',  'wb') as f:
    pickle.dump(model, f)
with open('tfidf.pkl',  'wb') as f:
    pickle.dump(tfidf, f)

print("Model saved as model.pkl and tfidf.pkl")
print("Run app.py next!")