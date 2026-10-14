# Q-A Cyclicity Hallucination Detector

This repository is a runnable pilot of the reference-free method agreed in the
project discussion. For an original question `Q` and initial answer `A0`, it
estimates four signals:

1. **Forward semantic entropy**: sample answers from `p(A | Q)`, cluster them,
   and measure how dispersed the answer distribution is.
2. **Cycle surprisal**: sample reconstructed questions from the answer alone,
   estimate the support `S_Q` for the original question, and compute
   `-log(S_Q)`.
3. **Competing-question margin**: compare the strongest alternative question
   cluster with the original-question support.
4. **InterrogateLLM similarity**: average the continuous cosine similarities
   between the original and reconstructed questions. Its risk form is
   `1 - mean_similarity`, so larger values indicate greater mismatch.

A calibrated logistic regression learns the final hallucination probability:

```text
sigmoid(b + w1 * H_forward + w2 * cycle_surprisal + w3 * competing_margin)
```

Raw inverse semantic entropy is also recorded for analysis, but it is not part
of the default classifier because it measures answer ambiguity and need not
increase with hallucination risk.

The ablation runner reports both the raw InterrogateLLM-style risk and a
logistic-calibrated version, as well as its combination with forward entropy.

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

The matched InterrogateLLM baseline uses the same reconstructed questions and
embedding model as the cycle detector:

```text
I_similarity = mean_k cosine(E(Q), E(Q'_k))
I_risk = 1 - I_similarity
```

Using identical samples isolates the effect of continuous mean similarity from
the thresholded support used by cycle surprisal.

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

## Slurm experiment using the existing HaloScope environment

The cluster pipeline reuses `$HOME/LLM_Haloscope/.venv`, the existing `HF_HOME`,
and a small dependency overlay. It converts the saved PEFT TriviaQA predictions,
extracts features in resumable atomic shards, trains the detector, and evaluates
validation and test splits in one self-requeuing job.

From the cloned repository, set the environment paths:

```bash
export HALOSCOPE_DIR="$HOME/LLM_Haloscope"
export HALOSCOPE_PYTHON="$HALOSCOPE_DIR/.venv/bin/python"
export CYCLE_OVERLAY="${SCRATCH:-$HOME}/qa-cycle-overlay"
export PYTHONPATH="$PWD:$CYCLE_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"

# Reuse the exact cache used by HaloScope/PEFT. Do not choose a new directory.
export HF_HOME=/absolute/path/to/existing/huggingface/cache

# Existing PEFT output; `correct` is converted to `label = 1 - correct`.
export SOURCE_PREDICTIONS=/absolute/path/to/outputs/llama2_7b_memory_probe/results/trivia_qa/predictions.jsonl
```

Run a 50-example smoke experiment first:

```bash
export CYCLE_RUN_ROOT="${SCRATCH:-$HOME}/qa-cycle-results/smoke-50"
export MAX_EXAMPLES=50
export SHARD_SIZE=5
bash slurm/submit_pipeline.sh
```

Monitor it with the job ID printed by the submission script:

```bash
squeue -j JOB_ID
tail -f slurm/logs/qa-cycle-JOB_ID.out
```

After the smoke run succeeds, submit a 500-example pilot into a new output
directory:

```bash
export CYCLE_RUN_ROOT="${SCRATCH:-$HOME}/qa-cycle-results/pilot-500"
export MAX_EXAMPLES=500
export SHARD_SIZE=10
bash slurm/submit_pipeline.sh
```

Use `MAX_EXAMPLES=0` for every eligible example. Always select a new
`CYCLE_RUN_ROOT` when changing the sample count or model configuration. A timed
out or preempted job requeues automatically; manually submitting the same run
again also skips completed feature shards.

Final outputs are:

```text
$CYCLE_RUN_ROOT/
├── data/{train,validation,test}.jsonl
├── features/{train,validation,test}/shard_*.jsonl
├── models/{forward_only,cycle_only,competition_only,interrogate_calibrated,
│          forward_cycle,forward_interrogate,full}.json
├── validation_metrics.json
└── test_metrics.json
```

The metrics files also contain `interrogate_raw`, which needs no trained model.
New extraction runs calculate these features automatically.

## Add InterrogateLLM scores to an existing completed run

Runs created before the InterrogateLLM feature was added can reuse their saved
inverse-question samples. This step runs only the embedding model and does not
generate any new answers or questions:

```bash
python -m scripts.backfill_interrogate \
  --run-root "$CYCLE_RUN_ROOT" \
  --embedding-model "$EMBEDDING_MODEL" \
  --device cpu \
  --local-files-only
```

The original shards remain unchanged. Augmented copies are written under
`$CYCLE_RUN_ROOT/features_interrogate`. Train and evaluate the expanded
ablation set with:

```bash
python -m scripts.run_ablations train \
  --input "$CYCLE_RUN_ROOT/features_interrogate/train" \
  --output-dir "$CYCLE_RUN_ROOT/models_interrogate"

python -m scripts.run_ablations evaluate \
  --input "$CYCLE_RUN_ROOT/features_interrogate/validation" \
  --model-dir "$CYCLE_RUN_ROOT/models_interrogate" \
  --output "$CYCLE_RUN_ROOT/validation_metrics_interrogate.json"

python -m scripts.run_ablations evaluate \
  --input "$CYCLE_RUN_ROOT/features_interrogate/test" \
  --model-dir "$CYCLE_RUN_ROOT/models_interrogate" \
  --output "$CYCLE_RUN_ROOT/test_metrics_interrogate.json"
```

For the full dataset on the configured cluster, run the resumable GPU version:

```bash
sbatch --export=ALL \
  --output=slurm/logs/interrogate-%j.out \
  --error=slurm/logs/interrogate-%j.err \
  slurm/interrogate_comparison.sbatch
```
