# Diagnosing and Improving Numerical Reasoning in Large Language Models

**CS 4650 — Final Project Report**

## 1. Introduction

Numerical reasoning remains a significant challenge for large language models (LLMs). While recent models have shown impressive performance on many NLP benchmarks, grade-school math word problems — which require multi-step arithmetic and logical reasoning — continue to expose fundamental limitations (Cobbe et al., 2021). Understanding *why* models fail at these tasks is arguably more valuable than simply reporting accuracy numbers, since it reveals what capabilities are missing and where future work should focus.

In this project, we evaluate several approaches to numerical reasoning on the GSM8K benchmark (Cobbe et al., 2021), a dataset of 8,792 grade-school math word problems. We compare simple baselines against zero-shot and chain-of-thought (CoT) prompting strategies (Wei et al., 2022), apply a lightweight verification step, and perform a detailed error analysis to categorize failure modes. Our main contribution is empirical: we develop an error taxonomy and use it to characterize how a small language model (GPT-2) fails at numerical reasoning, finding that the dominant failure mode is quantity misinterpretation rather than arithmetic errors.

## 2. Related Work

**GSM8K and math reasoning benchmarks.** Cobbe et al. (2021) introduced GSM8K as a benchmark specifically targeting multi-step mathematical reasoning. Unlike earlier arithmetic datasets, GSM8K problems require 2–8 reasoning steps and test compositional understanding rather than single-operation computation.

**Chain-of-thought prompting.** Wei et al. (2022) demonstrated that prompting LLMs to produce intermediate reasoning steps ("chain-of-thought") dramatically improves performance on reasoning tasks. However, this effect is primarily observed in large models (100B+ parameters); smaller models show limited benefit from CoT prompting.

**Verification and self-consistency.** Wang et al. (2023) proposed self-consistency decoding, where multiple reasoning paths are sampled and the most common answer is selected. Lighter-weight verification approaches include re-prompting the model to check its own answer, though these are less studied for small models.

**Error analysis in numerical reasoning.** Prior work has categorized errors in mathematical reasoning into arithmetic mistakes, logical errors, and misunderstanding of the problem statement (Lightman et al., 2023). Our work follows this line by applying a structured taxonomy to GPT-2's failures on GSM8K.

## 3. Methods

### 3.1 Dataset

We use GSM8K (Cobbe et al., 2021), loaded via HuggingFace Datasets. The dataset contains 7,473 training and 1,319 test examples. Each example consists of a natural language math question and a solution with step-by-step reasoning followed by a final numeric answer delimited by `####`. We evaluate on subsets of the test set (n=100 for prompting experiments, n=200 for baselines).

### 3.2 Baselines

We implement two simple baselines to establish lower bounds:

- **Majority baseline**: predicts the most common numeric answer from the training set (4.0, appearing 210 times) for every test example. This is analogous to the majority-class baseline standard in classification tasks.
- **Random baseline**: samples a random answer from the training answer distribution for each test example (seed=42 for reproducibility).

### 3.3 Prompting Strategies

We evaluate two prompting strategies using GPT-2 (124M parameters), a small causal language model:

- **Zero-shot**: the model receives the question with a direct instruction to solve it and provide the final numeric answer.
- **Chain-of-thought (CoT)**: the model receives the question with an instruction to reason step by step before giving its answer, following Wei et al. (2022).

We use greedy decoding (temperature=0) with a maximum of 150 new tokens per example.

### 3.4 Verification

We implement two lightweight verification strategies applied as post-processing on the model's predictions:

- **Heuristic verification**: flags answers that fail sanity checks — no valid number extracted, negative values, extreme magnitude (>1M), or non-integer answers (since most GSM8K answers are integers).
- **Re-prompt verification**: asks the model a second time whether its proposed answer is correct, giving it a chance to self-correct.

### 3.5 Evaluation

We use exact-match accuracy: the predicted answer (extracted as the last number in the model output) must equal the gold answer within a floating-point tolerance of 1e-5. We handle comma-separated numbers (e.g., "1,234") and decimal values.

### 3.6 Error Taxonomy

We categorize incorrect predictions into four types using automated heuristics, manually verified on a sample:

- **Arithmetic error**: the predicted value is within 25% of the gold answer, suggesting the model set up the problem correctly but miscalculated.
- **Multi-step reasoning failure**: the prediction is far from the gold answer, indicating missed or confused reasoning steps.
- **Quantity misinterpretation**: the predicted value matches a number appearing in the question text, suggesting the model extracted a quantity from the input rather than computing an answer.
- **Other**: no valid number was extracted from the model output.

## 4. Results

### 4.1 Overall Accuracy

| Method | n | Correct | Accuracy |
|--------|---|---------|----------|
| Majority baseline | 200 | 4 | 2.00% |
| Random baseline | 200 | 3 | 1.50% |
| Zero-shot (GPT-2) | 100 | 2 | 2.00% |
| CoT (GPT-2) | 100 | 2 | 2.00% |

All methods perform near chance level. GPT-2's accuracy matches the majority baseline, indicating that the model is not meaningfully reasoning about the math problems. Notably, CoT prompting does not improve over zero-shot for GPT-2, consistent with Wei et al.'s (2022) finding that CoT benefits are largely confined to models above 100B parameters.

### 4.2 Verification Results

Heuristic verification flagged 13/100 zero-shot predictions and 6/100 CoT predictions. The lower flag rate for CoT suggests that CoT outputs are more likely to contain plausible-looking numbers, even when incorrect. Neither verification strategy changed overall accuracy, since flagging an answer without a better alternative does not help (and the re-prompt strategy with GPT-2 does not produce better answers).

### 4.3 Error Analysis

| Error Type | Zero-Shot | CoT |
|------------|-----------|-----|
| Quantity misinterpretation | 71 (72.4%) | 69 (70.4%) |
| Multi-step reasoning failure | 16 (16.3%) | 23 (23.5%) |
| Other (no valid output) | 10 (10.2%) | 4 (4.1%) |
| Arithmetic error | 1 (1.0%) | 2 (2.0%) |

The dominant failure mode for GPT-2 is **quantity misinterpretation**: in over 70% of errors, the model outputs a number that appears verbatim in the question rather than computing a new value. For instance, given "She sells the remainder at the farmers' market daily for $2 per fresh duck egg," GPT-2 outputs "$2.50 per duck egg" — it echoes quantities from the input rather than performing any arithmetic.

This contrasts with the baselines, where multi-step reasoning failure dominates (75–89%) because the baseline predictions bear no relation to the question at all.

**CoT reduces "other" errors.** CoT prompting cuts the rate of unparseable outputs from 10.2% to 4.1%, suggesting the step-by-step framing helps the model at least produce numeric-looking text. However, it does not reduce quantity misinterpretation — the model still overwhelmingly copies numbers from the question rather than reasoning.

**Qualitative examples.** Inspecting GPT-2's outputs reveals a consistent pattern: the model tends to repeat the question or produce a surface-level rephrasing, then output one of the numbers from the question. In CoT mode, it sometimes generates formulas involving scientific notation (e.g., "1.5 × 10^-1.5") that bear no relation to the problem. The rare correct predictions (e.g., predicting 15 when the answer is 15) appear to be cases where the answer happens to match a quantity in the question.

## 5. Discussion

Our results confirm that GPT-2, as a 124M-parameter autoregressive LM, does not perform numerical reasoning on GSM8K. This is expected — GPT-2 was trained on web text and has no special mathematical training. The value of our experiments lies in the error analysis.

**Key finding: the bottleneck is understanding, not arithmetic.** Only 1–2% of errors are arithmetic mistakes. The vast majority are quantity misinterpretation (the model doesn't understand *what* to compute) or multi-step reasoning failures (the model can't chain operations). This suggests that for small models, the primary barrier to numerical reasoning is not calculation ability but rather compositional understanding of the problem structure.

**CoT is necessary but not sufficient.** CoT prompting improves output quality (fewer unparseable responses) but does not improve accuracy for small models. This is consistent with the hypothesis that CoT works by activating latent reasoning capabilities that simply don't exist in 124M-parameter models.

**Limitations.** Our study is limited to GPT-2. The error distribution would likely look very different for larger models (e.g., GPT-3.5, GPT-4, or LLaMA-70B), where CoT prompting has been shown to dramatically improve accuracy. Additionally, our error taxonomy is based on automated heuristics with manual spot-checks rather than exhaustive human annotation.

## 6. Conclusion

We evaluated majority-class baselines, zero-shot prompting, chain-of-thought prompting, and lightweight verification on GSM8K using GPT-2. All methods achieved approximately 2% accuracy. Our error analysis reveals that the dominant failure mode is quantity misinterpretation (70%+ of errors), where the model echoes numbers from the question rather than computing answers. This finding suggests that improving numerical reasoning in small LLMs requires advances in compositional understanding, not just arithmetic capability.

## References

- Cobbe, K., et al. (2021). Training Verifiers to Solve Math Word Problems. *arXiv:2110.14168*.
- Lightman, H., et al. (2023). Let's Verify Step by Step. *arXiv:2305.20050*.
- Wang, X., et al. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. *ICLR 2023*.
- Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS 2022*.

## Contributions

*[Fill in team member names and contributions here — 1 page max]*

- **[Name 1]**: Implemented evaluation pipeline, baseline predictors, and error analysis notebook.
- **[Name 2]**: Implemented prompting strategies (zero-shot, CoT), model abstraction, and verification module.
- **[Name 3]**: Ran experiments, performed data exploration, and wrote the report.
- **[Name 4]**: [...]
