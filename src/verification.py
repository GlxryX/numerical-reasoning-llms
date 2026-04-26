import json
import argparse
import os
from typing import Dict, List

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluate import extract_predicted_answer, is_correct
from src.prompting import MockModel, HFModel


# Verification strategies

def _build_verification_prompt(question: str, proposed: object) -> str:
    return (
        f"A student was asked the following math question:\n\n"
        f"Question: {question}\n\n"
        f"The student's answer was: {proposed}\n\n"
        f"Is this answer correct? If yes, repeat the answer. "
        f"If no, solve the problem and give the correct numeric answer.\n"
        f"Final answer:"
    )


def _verify_reprompt(records: List[Dict], model) -> List[Dict]:
    """Re-prompt the model to double-check each prediction."""
    verified: List[Dict] = []
    for i, rec in enumerate(records):
        orig = rec.get("predicted_answer") or rec.get("prediction", "")
        prompt = _build_verification_prompt(rec["question"], orig)
        raw = model.generate(prompt)
        new_ans = extract_predicted_answer(raw)

        entry = dict(rec)
        entry["verification_strategy"] = "reprompt"
        entry["verification_prompt"] = prompt
        entry["verification_raw_output"] = raw
        entry["verified_answer"] = new_ans
        entry["original_predicted_answer"] = orig
        verified.append(entry)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(records)}] orig={orig} -> verified={new_ans}")
    return verified


def _verify_heuristic(records: List[Dict]) -> List[Dict]:
    """Flag predictions that look suspicious (no number, negative, huge, non-int).
    Doesn't change the answer — just tags it for error analysis."""
    verified: List[Dict] = []
    for rec in records:
        orig = rec.get("predicted_answer") or rec.get("prediction", "")
        val = extract_predicted_answer(str(orig))

        flags: List[str] = []
        if val is None:
            flags.append("no_valid_number")
        else:
            if val < 0:
                flags.append("negative_answer")
            if abs(val) > 1_000_000:
                flags.append("extreme_magnitude")
            if val != int(val):
                flags.append("non_integer")

        entry = dict(rec)
        entry["verification_strategy"] = "heuristic"
        entry["heuristic_flags"] = flags
        entry["flagged"] = len(flags) > 0
        entry["verified_answer"] = val
        entry["original_predicted_answer"] = orig
        verified.append(entry)

    n_flagged = sum(1 for v in verified if v["flagged"])
    print(f"Heuristic verification: {n_flagged}/{len(verified)} flagged")
    return verified


# Main pipeline

def run_verification(input_path: str,
                     output_path: str = "results/verified_preds.json",
                     strategy: str = "heuristic",
                     mode: str = "mock",
                     model_name: str = "gpt2") -> List[Dict]:
    """Load predictions, run a verification pass, and save results.

    Args:
        input_path:  Path to a predictions JSON file.
        output_path: Where to save verified predictions.
        strategy:    'heuristic' or 'reprompt'.
        mode:        'mock' or 'hf' (only used for reprompt).
        model_name:  HF model id (only used for reprompt + hf).

    Returns:
        List of verified prediction dicts.
    """
    print(f"Loading predictions from {input_path}")
    with open(input_path) as f:
        records = json.load(f)
    print(f"  {len(records)} predictions loaded")

    if strategy == "reprompt":
        model = HFModel(model_name) if mode == "hf" else MockModel()
        print(f"Re-prompt verification with {model.name}")
        verified = _verify_reprompt(records, model)
    else:
        print("Running heuristic verification")
        verified = _verify_heuristic(records)

    # compare accuracy before/after
    total = len(records)
    orig_correct = sum(
        is_correct(str(r.get("raw_output", r.get("prediction", ""))), r["gold_answer"])
        for r in records
    )
    ver_correct = sum(
        is_correct(str(v["verified_answer"]), v["gold_answer"])
        for v in verified
    )
    print(f"\nOriginal accuracy:  {orig_correct}/{total} = {orig_correct/total:.4f}")
    print(f"Verified accuracy:  {ver_correct}/{total} = {ver_correct/total:.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(verified, f, indent=2)
    print(f"Saved -> {output_path}")
    return verified


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify GSM8K predictions")
    parser.add_argument("--input", required=True, help="Predictions JSON to verify")
    parser.add_argument("--output", default="results/verified_preds.json")
    parser.add_argument("--strategy", choices=["heuristic", "reprompt"], default="heuristic")
    parser.add_argument("--mode", choices=["mock", "hf"], default="mock")
    parser.add_argument("--model_name", default="gpt2")
    args = parser.parse_args()
    if not os.path.exists(args.input):
        print(f"File not found: {args.input}")
    else:
        run_verification(args.input, args.output, args.strategy,
                         args.mode, args.model_name)
