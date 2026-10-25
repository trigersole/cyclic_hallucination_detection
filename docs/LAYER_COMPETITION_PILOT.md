# Question-conditioned answer competition across layers

This is a separate, exploratory pilot. It does not modify the cycle detector, the
wording-stability pilot, model weights, or their result folders. No correctness
labels, reference answers, classifier fitting, or tuned thresholds are used.

## Cluster usage

From the repository on the cluster, after these files have been fetched:

```bash
python scripts/submit_layer_competition.py --dry-run
python scripts/submit_layer_competition.py
```

The launcher reuses the existing HaloScope Python environment, overlay, and model
cache. It defaults to Llama-2-7b-chat, MGPU-TC2, one GPU, 32 GB, and 90 minutes.
`--python`, `--model`, `--revision`, `--account`, and `--partition` override those
settings. Downloads are disabled unless `--allow-downloads` is passed. Results go
under `${SCRATCH:-$HOME}/qa-layer-results/layers-TIMESTAMP-ID`. Slurm logs are
`slurm/logs/layer-competition-JOBID.out` and `.err`. The launcher prints the exact
folder and job ID. To resume, repeat the same command and options with
`--run-root /exact/original/folder`. Requests already saved are reused. A completed
resume loads no model. Do not run two jobs against the same result folder.

When finished, get a compact report suitable for sharing:

```bash
python scripts/summarize_layer_competition.py "$RUN"
```

Here `$RUN` is the exact output folder printed by the launcher.

The worker also supports direct invocation and extra budget controls:

```bash
python -m qa_cycle_detector.layer_competition_cli \
  --run-root /new/separate/folder --local-files-only \
  --samples 4 --max-candidates 8 --max-new-tokens 64
```

## Experiment

Each question has manually specified controls in
`data/layer_competition_questions.jsonl`: one paraphrase, two changes to the key
detail, and one deletion of that detail. These questions contain no reference
answers. They reuse the four diagnostic topics; this is not a representative
benchmark. Manual controls need review when adding new data. The ablated questions
are intentionally underspecified; changing enzyme substrates can change relation
complexity as well as the identifying fact. Controls do not establish causality.

1. Generate one greedy answer and four stochastic answers to the original
   question. Generate one greedy answer to each control. Temperature is 0.9.
2. Extract contiguous answer phrases with the same generator. Invalid extraction,
   empty output, and truncation remain visible; none is treated as a factual error.
   Deduplicate only by the existing Unicode/case/edge-punctuation normalization.
   Aliases and qualifiers can still create distinct candidates. Keep at most eight
   candidates, retaining the greedy original answer first. Save all provenance and
   budget omissions.
3. Freeze candidate text and token IDs across conditions. Teacher-force each
   candidate under each question using the same short-answer instruction. Observe
   every decoder block's output at answer prediction positions. Project through
   the model's final norm and output head: an untrained logit lens. No hidden-state
   perturbations are applied. Layer IDs are one-based decoder block depths; the
   embedding layer is excluded.
4. Save each candidate's token log probabilities at each layer, excluding empty
   leading formatting tokens from reductions. Report three reductions separately:
   mean content-token log likelihood, summed content-token log likelihood, and
   first nonempty token log likelihood. Punctuation inside the extracted span can
   remain; raw tokens make this auditable. EOS is not scored. A candidate that is a
   prefix of another is consequently not a separate complete-string probability.
5. For each candidate and layer calculate original-minus-control log likelihood.
   Compare these gains across candidates, separately for wording, edited-detail,
   and ablated controls. The mean gain against edited questions is called detail
   specificity. Its target margin is the original greedy answer's gain minus the
   largest other candidate's gain. Positive values mean relative question-specific
   support, not factual correctness.

The primary `original_pool` contains only candidates obtained from original-question
sampling. A pool with fewer than two distinct candidates is explicitly invalid;
the method cannot claim certainty from a missing competitor. `expanded_pool` adds
candidates obtained from controls, and is a secondary analysis. Their source makes
them naturally suited to the edited questions, which can bias specificity; keep
the two analyses separate. Missing the correct candidate remains possible in both.

Outputs include original-question rankings as a baseline, paired-control rankings,
per-layer target margins, strict-win fractions, winner switches, first layer from
which the target remains strictly ahead, and mean target margin over the final
third of layers. These rules are fixed in advance. They do not produce a binary
hallucination label or a calibrated probability. Compare mean, sum, and first-token
results rather than choosing whichever looks best after seeing correctness.
Tied winners are recorded as `-1`; switches include transitions into or out of ties.

## Integrity and limitations

The final projected layer must match the decoder's already-normalized final
output at the scored token log probabilities, within 0.001 absolute error, or the
run fails. Hooks are removed even on exceptions. Only small selected-position
states are copied to CPU; the worker avoids full prompt vocabulary readouts.
Supported architectures are Llama, Qwen2, and Qwen3 exposing `model.layers`, a final
`norm`, and the ordinary output head. No learned tuned lens is fitted, so early
layer probabilities are uncalibrated readouts, not independent model beliefs.

Every candidate uses its own teacher-forced prefix. Later subwords can be
self-reinforcing; first-content-token results are provided as a check but can tie
when candidates share a first subword. Whole-answer scores have length biases.
Specificity can be strong for a consistently wrong association. A held-out
correctness evaluation is still required to establish detection performance;
those evaluation labels must not choose layers, candidates, or score formulas.

Cached requests, extracted phrases, and raw probes are saved under `examples/`.
`report.json` contains per-example summaries; `backend.json` records the resolved
model/runtime. Resume rejects changed settings or a changed resolved runtime when
new model requests are needed. Hard-killed runs may leave `.stability.lock` (shared
locking helper); remove it only after confirming the old process has stopped.

## Relation to prior work

Layer trajectories themselves are established, not a novelty claim:

- Jiang et al., [NAACL 2024](https://aclanthology.org/2024.naacl-long.60/):
  answer-token probability dynamics and hallucination classification.
- [DoLa](https://arxiv.org/abs/2309.03883): earlier/later layer contrasts for decoding.
- [TriLens](https://arxiv.org/abs/2606.01033): entropy trajectories across modules.
- Kim et al., [2025/2026](https://arxiv.org/abs/2507.06722): evidence that correct and
  incorrect outputs can exhibit similar layer-wise inference dynamics.

The proposed extension is the paired question-detail contrast of a frozen
candidate set across depth, with original-only candidate coverage and separate
wording/ablation controls. Its novelty and detection effectiveness are unverified.
