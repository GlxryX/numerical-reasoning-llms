import re
import json
import argparse
import os
from typing import Dict, List, Optional


# Answer extraction

def extract_predicted_answer(text: Optional[str]) -> Optional[float]:
    """Pull the last number from a model output string.
    Handles commas (e.g. '1,234') and decimals.
    """
    if text is None:
        return None
    text = str(text)
    nums = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?", text)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


def extract_ground_truth(text: Optional[str]) -> Optional[float]:
    """Extract gold answer from GSM8K's '#### <number>' format.
    Falls back to pulling the last number if there's no #### delimiter.
    """
    if text is None:
        return None
    text = str(text)
    if "####" in text:
        after = text.split("####")[-1].strip().replace(",", "")
        try:
            return float(after)
        except ValueError:
            return None
    return extract_predicted_answer(text)


# Scoring

def is_correct(pred_text: str, gold_text: str, tol: float = 1e-5) -> bool:
    """True when the predicted number matches the gold number within tolerance."""
    p = extract_predicted_answer(pred_text)
    g = extract_ground_truth(gold_text)
    if p is None or g is None:
        return False
    return abs(p - g) < tol


def exact_match_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """Exact-match accuracy over parallel lists of raw text."""
    if not predictions or len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must be non-empty and same length")
    correct = sum(is_correct(p, g) for p, g in zip(predictions, ground_truths))
    return correct / len(predictions)


# File-level evaluation

def evaluate_predictions_file(path: str) -> Dict:
    """Load a predictions JSON and compute + print accuracy.

    Args:
        path: Path to a JSON list of dicts. Each dict should have
              'raw_output' (or 'prediction') and 'gold_answer'.

    Returns:
        Dict with 'accuracy', 'total', 'correct', and per-example 'details'.
    """
    with open(path) as f:
        data = json.load(f)

    correct_count = 0
    details = []
    for i, entry in enumerate(data):
        pred_text = entry.get("raw_output") or entry.get("prediction", "")
        gold_text = entry.get("gold_answer", "")
        match = is_correct(pred_text, gold_text)
        if match:
            correct_count += 1
        details.append({
            "index": i,
            "predicted_value": extract_predicted_answer(pred_text),
            "gold_value": extract_ground_truth(gold_text),
            "correct": match,
        })

    acc = correct_count / len(data) if data else 0.0
    print(f"Evaluated {len(data)} examples from {path}")
    print(f"Correct: {correct_count}/{len(data)}")
    print(f"Accuracy: {acc:.4f}")
    return {"accuracy": acc, "total": len(data), "correct": correct_count, "details": details}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GSM8K predictions")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON")
    args = parser.parse_args()
    if not os.path.exists(args.predictions):
        print(f"File not found: {args.predictions}")
    else:
        evaluate_predictions_file(args.predictions)
