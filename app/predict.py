import joblib
import pandas as pd
import numpy as np

from feature_engineering import extract_custom_features

model = joblib.load("models/phishing_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict_email(email_text):

    tfidf = vectorizer.transform([email_text]).toarray()

    custom = pd.DataFrame(
        [extract_custom_features(email_text)]
    ).values

    X = np.hstack((tfidf, custom))

    prediction = model.predict(X)[0]

    return prediction


if __name__ == "__main__":

    email = input("Enter email content: ")

    result = predict_email(email)

    print(f"\nPrediction: {result}")
