import json
import argparse
import os
import random
from typing import Dict, List, Optional
from google import genai
from datasets import load_dataset
from dotenv import load_dotenv

import sys
load_dotenv()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluate import extract_predicted_answer, is_correct

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

class MockModel:
    """Returns random numbers so the pipeline can run without a GPU."""
    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self.name = "mock"

    def generate(self, prompt: str) -> str:
        return f"I think the answer is {self._rng.randint(1, 200)}."

class HFModel:
    """Thin wrapper around a HuggingFace causal LM (e.g. GPT-2)."""
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
            out = self.model.generate( # type: ignore
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)


class GeminiModel:
    """Wrapper for the Gemini API using the new google-genai SDK."""
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        print(f"Loading model: {model_name} ...")
        self.client = genai.Client()
        self.model_name = model_name
        self.name = "gemini"

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text if response.text is not None else "Error: Empty response"
            
        except Exception as e:
            print(f"Gemini Error: {e}")
            return "Error"

def run_prompting(mode: str = "mock",
                  prompt_type: str = "zero_shot",
                  subset_size: int = 50,
                  output_path: str = "results/prompted_preds.json",
                  model_name: str = "gpt2") -> List[Dict]:
    """Prompt a model on a GSM8K test subset and save predictions."""
    
    print(f"Config: mode={mode}, prompt_type={prompt_type}, n={subset_size}")

    ds = load_dataset("gsm8k", "main")
    test = ds["test"]
    n = min(subset_size, len(test))
    test_sub = test.select(range(n))

    if mode == "hf":
        model = HFModel(model_name)
    elif mode == "gemini":
        model = GeminiModel(model_name)
    else:
        model = MockModel()
        
    print(f"Using model: {model.name}")

    build_prompt = _PROMPT_BUILDERS[prompt_type]

    records: List[Dict] = []
    for i, ex_raw in enumerate(test_sub):
        ex = dict(ex_raw)
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
    parser.add_argument("--mode", choices=["mock", "hf", "gemini"], default="gemini",
                        help="mock = placeholder; hf = HuggingFace; gemini = Gemini API")
    parser.add_argument("--prompt_type", choices=["zero_shot", "cot"], default="cot")
    parser.add_argument("--subset_size", type=int, default=20)
    parser.add_argument("--output", default="results/cot_gemini.json")
    parser.add_argument("--model_name", default="gemini-2.5-flash",
                        help="Model id for HF or Gemini")
    args = parser.parse_args()

    run_prompting(args.mode, args.prompt_type, args.subset_size, args.output, args.model_name)