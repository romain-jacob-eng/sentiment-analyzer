# Project 4 — Sentiment Analyzer
# Loads a labeled text dataset, trains a TF-IDF + Logistic Regression model,
# and predicts whether a given review is positive or negative.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def load_data():
    data = {
        "text": [
            "Really held my attention the whole way through.",
            "Better than expected. Genuinely enjoyable.",
            "Solid portions, fair price. Will be back.",
            "Does exactly what it says. No complaints.",
            "Quiet, well-run, good food. Hard to fault.",
            "Great build quality. Feels like it'll last.",
            "One of the best I've seen in years.",
            "Absolutely loved it. Would recommend to anyone.",
            "Exceeded all my expectations.",
            "Fantastic experience from start to finish.",
            "Will definitely be coming back.",
            "Impressive quality for the price.",
            "Smooth, intuitive, and well-designed.",
            "Left feeling genuinely satisfied.",
            "Exactly what I was looking for.",
            "Outstanding. Hard to find fault with anything.",
            "A pleasant surprise. Very well done.",
            "Reliable, fast, and easy to use.",
            "Loved every minute of it.",
            "Top quality. Worth every penny.",
            "Food was fine. Service was not.",
            "Stopped working after three weeks.",
            "Too long, and the ending goes nowhere.",
            "Waited 45 minutes for a lukewarm dish.",
            "Setup took forever and the app keeps crashing.",
            "Decent premise, poor execution.",
            "Not worth the price at all.",
            "Disappointing from start to finish.",
            "Would not recommend to anyone.",
            "Broke after one use. Total waste of money.",
            "Slow, buggy, and frustrating to use.",
            "The worst experience I've had in a long time.",
            "Overcrowded, overpriced, and understaffed.",
            "Nothing worked as advertised.",
            "Left feeling completely let down.",
            "Poor quality. Fell apart immediately.",
            "Avoid at all costs.",
            "Terrible service. Never going back.",
            "Confusing interface and constant errors.",
            "A complete waste of time and money."
        ],
        "label": [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    }
    df = pd.DataFrame(data)
    return df


def train_model(df):
    X = df["text"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    return vectorizer, model, X_test_tfidf, y_test


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    classification = classification_report(y_test, predictions)
    print(f"Model accuracy: {accuracy * 100:.1f}%")
    print("Classification Report:")
    print(classification)


def predict_sentiment(text, vectorizer, model):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    if prediction[0] == 1:
        return "Positive"
    else:
        return "Negative"


if __name__ == "__main__":
    df = load_data()
    vectorizer, model, X_test, y_test = train_model(df)
    evaluate_model(model, X_test, y_test)
    result = predict_sentiment("It was absolutely fantastic", vectorizer, model)
    print(f"Sentiment: {result}")
