"""Optional fine-tuning scaffold — not used in main experiments.

Full fine-tuning is out of scope (compute + time). This just has a data-prep
utility and a stub showing how you'd extend it.
"""

import json
import os


def prepare_finetuning_data(output_path: str = "results/finetuning_data.jsonl") -> None:
    """Convert GSM8K train split to input/target JSONL."""
    from datasets import load_dataset

    train = load_dataset("gsm8k", "main")["train"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for ex_raw in train:
            ex = dict(ex_raw)
            line = {"input": f"Solve: {ex['question']}", "target": ex["answer"]}
            f.write(json.dumps(line) + "\n")
    print(f"Wrote {len(train)} examples -> {output_path}")


def finetune_model() -> None:
    """Stub — would fine-tune a seq2seq model on GSM8K.

    Sketch:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
        tok = AutoTokenizer.from_pretrained("t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        # tokenize, set TrainingArguments, call trainer.train()
    """
    raise NotImplementedError("Fine-tuning not implemented")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare_data", action="store_true",
                        help="Export GSM8K training data to JSONL")
    args = parser.parse_args()
    if args.prepare_data:
        prepare_finetuning_data()
    else:
        print("fine_tuning.py — stub (not used in main experiments)")
        print("Run with --prepare_data to export training data.")
