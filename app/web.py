from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

from feature_engineering import extract_custom_features

app = Flask(__name__)

# Load trained model
model = joblib.load("models/phishing_model.pkl")

# Load vectorizer
vectorizer = joblib.load("models/vectorizer.pkl")


def predict_email(email_text):

    # TF-IDF Features
    tfidf = vectorizer.transform([email_text]).toarray()

    # Custom Features
    custom = pd.DataFrame(
        [extract_custom_features(email_text)]
    ).values

    # Combine Features
    X = np.hstack((tfidf, custom))

    # Prediction
    prediction = model.predict(X)[0]

    # Confidence Score
    prediction_proba = model.predict_proba(X)[0]
    score = round(max(prediction_proba) * 100, 2)

    return prediction, score


@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    score = None
    email_text = ""

    if request.method == "POST":

        email_text = request.form["email"]

        result, score = predict_email(email_text)

    return render_template(
        "index.html",
        result=result,
        score=score,
        email=email_text
    )


if __name__ == "__main__":
    app.run(debug=True)
