import json
import argparse
import os
from collections import Counter
from typing import List

from datasets import load_dataset

from src.evaluate import extract_ground_truth, exact_match_accuracy


# --- Baseline strategies ---

def _majority_baseline(train_answers: List[float], n: int) -> List[str]:
    """Always predict the single most common answer from training."""
    counts = Counter(train_answers)
    most_common = counts.most_common(1)[0][0]
    print(f"Most frequent training answer: {most_common} ({counts[most_common]} occurrences)")
    return [str(most_common)] * n


def _random_baseline(train_answers: List[float], n: int, seed: int = 42) -> List[str]:
    """Randomly sample from the training answer distribution."""
    import random
    rng = random.Random(seed)
    return [str(rng.choice(train_answers)) for _ in range(n)]


# --- Main pipeline ---

def run_baseline(subset_size: int = 100,
                 output_path: str = "results/baseline_preds.json",
                 baseline_type: str = "majority") -> float:
    """Run a simple baseline predictor on a GSM8K test subset.

    Args:
        subset_size:   Number of test examples to evaluate.
        output_path:   Where to save the predictions JSON.
        baseline_type: 'majority' or 'random'.

    Returns:
        Exact-match accuracy as a float.
    """
    print("Loading GSM8K...")
    ds = load_dataset("gsm8k", "main")
    train, test = ds["train"], ds["test"]

    n = min(subset_size, len(test))
    test_sub = test.select(range(n))
    print(f"Evaluating on {n} test examples")

    # collect numeric answers from training split
    train_answers: List[float] = []
    for ex in train:
        val = extract_ground_truth(ex["answer"])
        if val is not None:
            train_answers.append(val)

    if baseline_type == "random":
        preds = _random_baseline(train_answers, n)
    else:
        preds = _majority_baseline(train_answers, n)

    golds = [ex["answer"] for ex in test_sub]
    acc = exact_match_accuracy(preds, golds)
    print(f"Baseline ({baseline_type}) accuracy: {acc:.4f}")

    # save predictions
    records = []
    for i, (ex, pred) in enumerate(zip(test_sub, preds)):
        records.append({
            "index": i,
            "question": ex["question"],
            "gold_answer": ex["answer"],
            "prediction": pred,
            "raw_output": pred,
            "method": f"baseline_{baseline_type}",
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} predictions -> {output_path}")
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GSM8K baselines")
    parser.add_argument("--subset_size", type=int, default=100)
    parser.add_argument("--output", default="results/baseline_preds.json")
    parser.add_argument("--baseline_type", choices=["majority", "random"], default="majority")
    args = parser.parse_args()
    run_baseline(args.subset_size, args.output, args.baseline_type)
