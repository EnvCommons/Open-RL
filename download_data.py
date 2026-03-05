#!/usr/bin/env python3
"""Download Open-RL dataset from HuggingFace and save locally."""

import json
from datasets import load_dataset


def main():
    print("Downloading Open-RL dataset from HuggingFace...")
    dataset = load_dataset("TuringEnterprises/Open-RL", split="train")

    data = []
    for row in dataset:
        data.append({
            "conversation_id": row["conversation_id"],
            "domain": row["domain"],
            "sub_domain": row["sub_domain"],
            "question": row["question"],
            "answer": row["answer"]
        })

    with open("data.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data)} tasks to data.json")

    # Print domain distribution
    from collections import Counter
    domains = Counter(row["domain"] for row in data)
    print("\nDomain distribution:")
    for domain, count in sorted(domains.items()):
        print(f"  {domain}: {count}")


if __name__ == "__main__":
    main()
