from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.baseline import load_claims


train_text, train_labels = load_claims("train.jsonl")

X_train, X_val, y_train, y_val = train_test_split(
    train_text,
    train_labels,
    test_size=0.2,
    random_state=42,
    stratify=train_labels
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))

for weighting in [None, "balanced"]:

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=2
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight=weighting
        ))
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_val)

    print(f"\nClass weight: {weighting}")
    print(classification_report(
        y_val,
        predictions,
        digits=4,
        zero_division=0
    ))