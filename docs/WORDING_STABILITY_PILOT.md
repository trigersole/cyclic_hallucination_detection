# Wording-corrected semantic stability: small unlabeled pilot

This pilot tests a candidate combination: estimate prompt-wording directions,
perturb a model inside/outside those directions, and measure changes in answer
meaning with matched random-subspace controls. This is a diagnostic experiment,
not a validated hallucination detector or a claim of established novelty.

No correctness labels, classifier training, label-based layer/rank selection, or
threshold fitting is used. The loader explicitly discards `label`, grading fields,
legacy features, and sample answers. Only `question`, optional supplied `answer`,
and optional `id` are retained. Label changes alone cannot change inputs, seeds,
or outputs. By default, the pilot uses the new question-only dataset at
`data/stability_pilot_questions.jsonl` and generates its own answers. It has twenty
manually composed questions, no reference answers or correctness labels, and no
dependency on old runs. It is a small convenience set, not a representative or
graded benchmark. Existing QA JSONL/shard directories remain optional inputs.

## Simplest Slurm setup

Once the new pilot files are in your cluster checkout, run from that repository:

```bash
python scripts/submit_stability.py
```

This submits a two-example smoke test using fresh bundled questions and newly
generated answers. It chooses the existing HaloScope Python
environment if available (or the invoking Python), preserves your existing cache
settings, verifies the model environment and locally cached snapshots, creates a
unique output directory, and prints the job ID, log command, and result path. It
does not download models by default. No GPU check is attempted on the login node.

For twenty examples:

```bash
python scripts/submit_stability.py --examples 20
```

Optional overrides, only if your environment needs them:

```bash
python scripts/submit_stability.py /path/to/input.jsonl \
  --python /path/to/existing/venv/bin/python --account YOUR_ACCOUNT
```

Use `--dry-run` to validate inputs and print the submission without creating a job
or result directories. For a custom shared cache, retain/export your existing
`HF_HOME`; the script cannot infer an unconfigured cache elsewhere on the cluster.
Use `--run-root /path/to/previous-pilot` with identical settings to resume, and
remove a stale lock only after checking that the previous job is no longer running.
The submitter prints a basic resume command; repeat any model/account overrides
used originally. The cache preflight checks snapshot availability; actual weight
loading and GPU compatibility are checked when the job runs.

## What is implemented

1. Keep the question verbatim. Use six fixed answer-instruction variants to estimate
   wording variation; no reconstructed questions or generated paraphrases are used.
2. Capture each prompt's final-token state at the output of one decoder block
   (default: the middle block, zero-based). Use anchor-relative displacements to
   obtain an SVD wording basis, default rank 3. Cap rank at numerical rank and
   hidden dimension minus one. Fail explicitly if no basis is identifiable.
3. Generate an unperturbed greedy anchor answer. When the input contains only a
   question, this becomes the answer being analyzed (`answer_source=generated_anchor`).
   No old predictions or answers are required. Separately sample four answers
   with temperature 0.8 as the ordinary sampling-entropy baseline (no top-p filter).
4. For each of three perturbation amplitudes and four noise directions, generate
   answers under five conditions:

   | Condition | Direction space |
   |---|---|
   | `isotropic` | Full hidden space |
   | `wording` | Estimated wording basis |
   | `orthogonal` | Its orthogonal complement |
   | `random_wording` | Random basis of the same rank |
   | `random_orthogonal` | Complement of that random basis |

   All five conditions use the same underlying Gaussian draw for a direction.
   Each projected direction is renormalized to exactly
   `amplitude * ||anchor_state||_2`. Directions are reused across amplitudes. Both
   complement conditions have the same dimension; both in-subspace conditions have
   the same dimension. The comparison does not silently reduce noise magnitude.
5. Apply the intervention to the **last prompt token only**, during prefill, then
   remove the hook before cached decoding. Perturbed decoding is greedy so that
   differences arise from the intervention, rather than output-sampling randomness.
   Model weights are never changed. The backend supports Llama/Qwen-style models
   exposing `model.model.layers`; it is not a generic adapter for all architectures.
6. Embed answers using the existing sentence encoder. Estimate semantic clusters
   and anchored meaning changes using cosine similarity with a fixed threshold
   (default 0.82). This is an approximate semantic comparison, not a factuality judge.
   Clustering uses the project's connected-components rule; chaining can merge
   imperfectly equivalent responses. Audit actual answers, not only entropy values.

With defaults, each example uses 6 capture forward passes and 65 generations:
1 greedy anchor + 4 sampled baseline answers + 5 conditions × 3 amplitudes × 4
directions. All perturbation conditions have identical generation/token limits.
Twenty examples allow at most 31,200 generated tokens; actual EOS termination
usually lowers that bound. Runtime is model- and hardware-dependent.

## Preview without loading models or writing files

```bash
python -m qa_cycle_detector.stability_cli \
  --input /absolute/path/to/existing/features/test \
  --run-root /absolute/path/to/new-stability-pilot \
  --max-examples 20 --dry-run
```

The first N input rows are selected in file/shard order without looking at labels.
For an initially smaller smoke run, use `--max-examples 2` and a separate root.
The dry run validates selected rows and configuration, and reports compute budget.
It does not reserve or validate the output directory.

## Run in an allocated GPU environment

```bash
python -m qa_cycle_detector.stability_cli \
  --input /absolute/path/to/existing/features/test \
  --run-root /absolute/path/to/new-stability-pilot-20 \
  --max-examples 20 \
  --model meta-llama/Llama-2-7b-chat-hf \
  --embedding-model sentence-transformers/all-mpnet-base-v2 \
  --embedding-device cpu \
  --rank 3 --directions 4 --amplitudes 0.005 0.02 0.05 \
  --baseline-samples 4 --max-new-tokens 24
```

Use the project's existing Torch/Transformers environment and shared model cache.
No new dependencies are added. Use `--local-files-only` when models are cached and
`--revision COMMIT` to pin generator weights. For the sentence encoder, an immutable
local snapshot path is preferable. Layer, precision, instruction variants, input
selection, and sampling settings are frozen in the run marker. Resolved model
commit/runtime information is recorded in `MODEL.json` and checked on generation
resume. Results can vary across hardware even with fixed seeds.

If using the existing HaloScope environment, replace `python` with
`"$HALOSCOPE_DIR/.venv/bin/python"` and retain the existing `HF_HOME` and dependency
overlay through `PYTHONPATH="$PWD:$CYCLE_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"`.

The independent Slurm launcher uses the same partition/GPU layout as the current
project but does not invoke or alter the cycle pipeline:

```bash
export STABILITY_INPUT=/absolute/path/to/existing-run/features/test
export STABILITY_RUN_ROOT="${SCRATCH:-$HOME}/qa-stability-results/pilot-20"
export STABILITY_MODEL=meta-llama/Llama-2-7b-chat-hf
export STABILITY_MAX_EXAMPLES=20
# Keep HF_HOME pointing to the existing shared cache.
sbatch --output=wording-stability-%j.out --error=wording-stability-%j.err \
  slurm/stability_pilot.sbatch
```

Add the appropriate `--account` option to `sbatch` if required. This creates a new
job/allocation; no existing jobs are canceled, requeued, or reused. A launcher has
been supplied, but no job is submitted by implementing or inspecting this pilot.

## Scores and interpretation

Per amplitude and condition, `scores.json` records:

- Approximate semantic entropy across generated answers.
- Fraction differing in meaning from the unperturbed greedy anchor.
- Fraction differing from the supplied answer (reported separately).
- Mean anchor similarity, predictive token entropy, empty-output rate, and
  truncation rate.

`flip_auc[condition]` is the trapezoidal integral of anchor flip fraction over the
tested relative amplitudes, divided by the maximum amplitude, with a zero-flip
point at amplitude zero. It is a sensitivity summary, **not AUROC**.

The proposed contrast is:

```text
wording_corrected = flip_auc[orthogonal] - flip_auc[wording]
random_corrected = flip_auc[random_orthogonal] - flip_auc[random_wording]
wording_specificity_contrast = wording_corrected - random_corrected
```

These are signed descriptive scores, not learned weights, correctness probabilities,
or calibrated risk estimates. A positive contrast is not by itself evidence of an
error: correct factual answers can be sensitive in semantic directions too.

`first_tested_flip_amplitude` is the first tested amplitude at which at least one
sample changes meaning. It is a finite-direction search observation, **not** a
certified radius or the minimum adversarial perturbation. A null result means no
observed flip or unusable observations; check `usable_conditions` to distinguish.

If any outputs in a condition are empty/truncated, or the anchor is empty/truncated,
that condition's integrated score is null rather than interpreting generation damage
as factual uncertainty. Raw diagnostics remain available. Increase the token limit
in a new pilot root if truncation is frequent; do not change settings mid-run.

The primary intervention measurements describe stability of the model's own greedy
answer to the question. If that anchor does not match the supplied answer,
`supplied_answer_score` is null; question-level sensitivity remains available. This
prevents silently treating stability of a newly generated answer as a score for a
different answer from an existing run. No answer is regraded or rewritten.
For question-only inputs, there is no supplied-answer comparison: the generated
anchor is the target. Output names containing `supplied` are retained for schema
compatibility and refer to that anchor in this mode; `answer_source` makes this
explicit. Empty/truncated anchors yield unusable scores, not factual judgments.

## Output and resume

```text
new-stability-pilot/
  STABILITY_RUN.json
  MODEL.json
  report.json
  examples/00000/
    views/0.json ... 5.json
    states.json
    generations/anchor.json, baseline_*.json, a*_d*_*.json
    raw.json
    scores.json
```

Run the same command to resume. Each capture, generation, and final score is
checkpointed atomically. A completed pilot loads neither the generator nor the
semantic encoder. Changed inputs/settings require a new directory. Existing
nonempty directories are never adopted. Directories marked as legacy extraction
or triplet outputs, and their descendants, are rejected.

The `.stability.lock` prevents concurrent writes. A hard kill may leave it behind;
remove only that file once no process is using the root, then resume. The launcher
does not automatically requeue. Stage timing and generated token counts (including
EOS) are retained from the original computation across resumes. GPU model loading
and tokenizer time are not separately benchmarked; generation elapsed time includes
tokenization and decoding. Encoder timing is reported separately.

## What this pilot can establish

First inspect answer outputs and corruption rates, not only score rankings. Check
whether the wording correction differs from equal-rank random controls, whether
the effect survives alternative fixed seeds, and whether it is driven by length or
instruction changes. Use a new root for each predeclared configuration. A single
20-example, four-direction run cannot establish detection performance or novelty.
Correctness labels can optionally be used later for a held-out audit, but are not
read by this pilot. There is no label-based detector training or threshold fitting.

Known limitations: instruction changes also alter token positions; the estimated
basis is not proven to contain only wording information; random internal movement
can damage computation; sentence similarity can miss entity substitutions; four
directions give noisy flip estimates; one random control basis is a limited null
control; and a stable false belief can evade the method.

Offline verification:

```bash
python -m pytest -q --basetemp=.pytest_cache/stability-tests
```

Tests cover equal perturbation norms, subspace membership, instruction preservation,
label independence, checkpoint resume, invalid-output handling, control-flow removal
of intervention hooks, and cached decoding. These tests use fake models/Torch-like
tensors and do not establish real GPU/Hugging Face compatibility or scientific
effectiveness. A real-model smoke run is still required on the GPU environment.
