import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

from feature_engineering import extract_custom_features

# Load dataset
df = pd.read_csv("data/emails.csv")

# Text and labels
X_text = df["text"]
y = df["label"]

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")

X_tfidf = vectorizer.fit_transform(X_text)

# Custom Features
custom_features = X_text.apply(extract_custom_features)
custom_df = pd.DataFrame(custom_features.tolist())

# Combine Features
X = np.hstack((X_tfidf.toarray(), custom_df.values))

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("reports/confusion_matrix.png")

# Classification Report
report = classification_report(y_test, y_pred)

with open("reports/classification_report.txt", "w") as f:
    f.write(report)

# Save model
joblib.dump(model, "models/phishing_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("[+] Model trained successfully")
print("[+] Confusion matrix saved")
print("[+] Classification report saved")
