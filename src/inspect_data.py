import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data/raw")


def inspect_file(filename):
    path = DATA_DIR / filename
    labels = Counter()
    examples = []
    total = 0

    with path.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            total += 1
            labels[record["label"]] += 1

            if len(examples) < 3:
                examples.append({
                    "claim": record["claim"],
                    "label": record["label"]
                })

    print(f"\nDataset: {filename}")
    print(f"Total claims: {total}")
    print(f"Class distribution: {dict(labels)}")

    for example in examples:
        print(f"\nClaim: {example['claim']}")
        print(f"Label: {example['label']}")


if __name__ == "__main__":
    inspect_file("train.jsonl")
    inspect_file("paper_dev.jsonl")