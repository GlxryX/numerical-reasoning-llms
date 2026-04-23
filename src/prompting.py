import json
import argparse
import os
import random
from typing import Dict, List, Optional

from datasets import load_dataset

from src.evaluate import extract_predicted_answer, is_correct


# --- Prompt templates ---

def _build_zero_shot_prompt(question: str) -> str:
    return (
        f"Solve the following math problem. "
        f"Give only the final numeric answer.\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def _build_cot_prompt(question: str) -> str:
    """Chain-of-thought prompt following Wei et al. (2022)."""
    return (
        f"Solve the following math problem step by step. "
        f"Show your reasoning, then give the final numeric answer "
        f"on the last line.\n\n"
        f"Question: {question}\n"
        f"Let's think step by step:\n"
    )


_PROMPT_BUILDERS = {
    "zero_shot": _build_zero_shot_prompt,
    "cot": _build_cot_prompt,
}


# --- Model wrappers ---

class MockModel:
    """Returns random numbers so the pipeline can run without a GPU."""
    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self.name = "mock"

    def generate(self, prompt: str) -> str:
        return f"I think the answer is {self._rng.randint(1, 200)}."


class HFModel:
    """Thin wrapper around a HuggingFace causal LM (e.g. GPT-2).
    GPT-2 isn't good at math — point is to run the pipeline and study errors."""

    def __init__(self, model_name: str = "gpt2", max_new_tokens: int = 150):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"Loading model: {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self._max_new_tokens = max_new_tokens
        self.name = model_name
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt",
                                truncation=True, max_length=512)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)


# --- Main pipeline ---

def run_prompting(mode: str = "mock",
                  prompt_type: str = "zero_shot",
                  subset_size: int = 50,
                  output_path: str = "results/prompted_preds.json",
                  model_name: str = "gpt2") -> List[Dict]:
    """Prompt a model on a GSM8K test subset and save predictions.

    Args:
        mode:        'mock' for placeholder outputs, 'hf' for a real HF model.
        prompt_type: 'zero_shot' or 'cot'.
        subset_size: Number of test examples.
        output_path: Where to write the predictions JSON.
        model_name:  HF model id (only used when mode='hf').

    Returns:
        List of per-example prediction dicts.
    """
    print(f"Config: mode={mode}, prompt_type={prompt_type}, n={subset_size}")

    ds = load_dataset("gsm8k", "main")
    test = ds["test"]
    n = min(subset_size, len(test))
    test_sub = test.select(range(n))

    model = HFModel(model_name) if mode == "hf" else MockModel()
    print(f"Using model: {model.name}")

    build_prompt = _PROMPT_BUILDERS[prompt_type]

    records: List[Dict] = []
    for i, ex in enumerate(test_sub):
        prompt = build_prompt(ex["question"])
        raw_out = model.generate(prompt)
        pred = extract_predicted_answer(raw_out)
        records.append({
            "index": i,
            "question": ex["question"],
            "gold_answer": ex["answer"],
            "prompt_type": prompt_type,
            "prompt": prompt,
            "raw_output": raw_out,
            "predicted_answer": pred,
        })
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{n}] pred={pred}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} predictions -> {output_path}")

    n_correct = sum(is_correct(r["raw_output"], r["gold_answer"]) for r in records)
    print(f"Accuracy ({prompt_type}, {model.name}): "
          f"{n_correct}/{len(records)} = {n_correct/len(records):.4f}")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt-based GSM8K evaluation")
    parser.add_argument("--mode", choices=["mock", "hf"], default="mock",
                        help="mock = placeholder outputs; hf = real HuggingFace model")
    parser.add_argument("--prompt_type", choices=["zero_shot", "cot"], default="zero_shot")
    parser.add_argument("--subset_size", type=int, default=50)
    parser.add_argument("--output", default="results/prompted_preds.json")
    parser.add_argument("--model_name", default="gpt2",
                        help="HF model id (only used with --mode hf)")
    args = parser.parse_args()
    run_prompting(args.mode, args.prompt_type, args.subset_size,
                  args.output, args.model_name)
