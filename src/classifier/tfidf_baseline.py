import json
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

LABELS_PATH = os.path.join("data", "golden_set", "labels_final.json")
MODEL_PATH = os.path.join("outputs", "tfidf_model.pkl")


def load_data(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    texts = [l["customer_text"] for l in data["labels"]]
    intents = [l["intent"] for l in data["labels"]]
    return texts, intents


def train(texts, intents):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True,
            min_df=2
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        texts, intents,
        test_size=0.2,
        random_state=42,
        stratify=intents
    )

    pipeline.fit(X_train, y_train)

    print("TF-IDF Baseline — Test Set Results:")
    print("-" * 50)
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))

    return pipeline, X_test, y_test


def save_model(pipeline, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to {path}")


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def classify(text: str, pipeline=None) -> str:
    if pipeline is None:
        pipeline = load_model(MODEL_PATH)
    return pipeline.predict([text])[0]


if __name__ == "__main__":
    texts, intents = load_data(LABELS_PATH)
    pipeline, X_test, y_test = train(texts, intents)
    save_model(pipeline, MODEL_PATH)