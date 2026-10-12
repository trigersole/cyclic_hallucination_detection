# Q-A Cyclicity Hallucination Detector

This repository is a runnable pilot of the reference-free method agreed in the
project discussion. For an original question `Q` and initial answer `A0`, it
estimates three signals:

1. **Forward semantic entropy**: sample answers from `p(A | Q)`, cluster them,
   and measure how dispersed the answer distribution is.
2. **Cycle surprisal**: sample reconstructed questions from the answer alone,
   estimate the support `S_Q` for the original question, and compute
   `-log(S_Q)`.
3. **Competing-question margin**: compare the strongest alternative question
   cluster with the original-question support.

A calibrated logistic regression learns the final hallucination probability:

```text
sigmoid(b + w1 * H_forward + w2 * cycle_surprisal + w3 * competing_margin)
```

Raw inverse semantic entropy is also recorded for analysis, but it is not part
of the default classifier because it measures answer ambiguity and need not
increase with hallucination risk.

## Important scope

This detector measures uncertainty and question-answer cycle mismatch. It does
not establish factual truth. A false answer that is both confidently repeated
and perfectly aligned with the question can evade any purely reference-free
consistency method. The intended first experiment is short-form, single-fact
factual QA.

## Installation

Python 3.10+ and a CUDA-capable GPU are recommended for the default 3B model.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

The default generator is `Qwen/Qwen2.5-3B-Instruct`; both generator and
embedding model are configurable. Model weights are downloaded by Hugging Face
on first use.

## Input format

Use JSON Lines with one object per example:

```json
{"id":"q1","question":"What is the capital of France?","answer":"Paris","label":0}
{"id":"q2","question":"What is the capital of France?","answer":"Rome","label":1}
```

`label=1` means hallucinated and `label=0` means correct. The supplied answer is
treated as `A0`; this keeps the experiment reproducible and allows evaluation
on benchmark-provided model answers.

## Run the pipeline

Extract and cache the expensive generation features:

```powershell
qa-cycle extract `
  --input data/example.jsonl `
  --output artifacts/features.jsonl `
  --keep-samples
```

Train the detector on a labeled training feature file:

```powershell
qa-cycle train `
  --input artifacts/train_features.jsonl `
  --output artifacts/detector.json
```

Evaluate on a separate labeled test split:

```powershell
qa-cycle evaluate `
  --input artifacts/test_features.jsonl `
  --model artifacts/detector.json `
  --output artifacts/test_metrics.json
```

Score one pair (with or without a trained classifier):

```powershell
qa-cycle score `
  --question "What is the capital of France?" `
  --answer "Rome" `
  --model artifacts/detector.json
```

Useful experiment controls include `--forward-samples`, `--inverse-samples`,
`--forward-temperature`, `--inverse-temperature`, and `--threshold`. Run
`qa-cycle <command> --help` for all options.

## Mathematical details

Let inverse samples form semantic clusters with `K` total samples. Samples
whose embeddings pass the equivalence threshold against the original question
count toward `n_Q`. With additive smoothing `lambda`,

```text
S_Q = (n_Q + lambda) / (K + lambda * c)
L_cycle = -log(S_Q)
```

Here `c` is the number of observed inverse clusters. If no generated question
matches the original, the original question is added as an unobserved category
for smoothing.

For the largest non-original cluster with count `n_alt`,

```text
p_alt = (n_alt + lambda) / (K + lambda * c)
R_Q = log(p_alt / S_Q)
```

Positive `R_Q` means an alternative question is more strongly supported than
the original. Semantic entropy uses natural logarithms.

## Experimental cautions

- Tune the equivalence threshold only on training/validation data.
- Validate semantic equivalence on a manually annotated question-pair subset;
  embedding similarity is a pilot approximation, not a gold-standard judge.
- Keep train, validation, and test examples separate before fitting logistic
  regression.
- Compare against forward entropy alone and report AUROC, AUPRC, confidence
  intervals, and performance on confidently-wrong examples.
- Record model name, prompt, temperature, random seed, `M`, and `K` for every
  experiment.

## Tests

The tests use fake generators and embeddings, so they do not download models:

```powershell
pytest -q
```
