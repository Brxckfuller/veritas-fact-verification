from pathlib import Path
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def load_claims(filename):
    claims = []
    labels = []

    with (DATA_DIR / filename).open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            claims.append(record["claim"])
            labels.append(record["label"])

    return claims, labels


def main():
    train_text, train_labels = load_claims("train.jsonl")
    dev_text, dev_labels = load_claims("paper_dev.jsonl")

    print("Training claims:", len(train_text))
    print("Development claims:", len(dev_text))

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=2
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    print("\nTraining baseline classifier...")
    pipeline.fit(train_text, train_labels)

    print("Evaluating on development set...")
    predictions = pipeline.predict(dev_text)

    print("\nAccuracy:", accuracy_score(dev_labels, predictions))

    print("\nClassification report:")
    print(classification_report(
        dev_labels,
        predictions,
        digits=4,
        zero_division=0
    ))

    print("\nConfusion matrix:")
    labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

    print("Label order:", labels)
    print(confusion_matrix(
        dev_labels,
        predictions,
        labels=labels
    ))


if __name__ == "__main__":
    main()