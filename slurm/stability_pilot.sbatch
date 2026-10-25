#!/usr/bin/env bash
#SBATCH --partition=MGPU-TC2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --job-name=wording-stability

# Separate job, inputs, outputs, and environment variables from the cycle pipeline.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
STABILITY_INPUT=${STABILITY_INPUT:-${SLURM_SUBMIT_DIR}/data/stability_pilot_questions.jsonl}
STABILITY_RUN_ROOT=${STABILITY_RUN_ROOT:?Set STABILITY_RUN_ROOT to a new separate directory}
STABILITY_MODEL=${STABILITY_MODEL:-meta-llama/Llama-2-7b-chat-hf}
STABILITY_EMBEDDING_MODEL=${STABILITY_EMBEDDING_MODEL:-sentence-transformers/all-mpnet-base-v2}
HALOSCOPE_DIR=${HALOSCOPE_DIR:-$HOME/LLM_Haloscope}
STABILITY_PYTHON=${STABILITY_PYTHON:-$HALOSCOPE_DIR/.venv/bin/python}
CYCLE_OVERLAY=${CYCLE_OVERLAY:-${SCRATCH:-$HOME}/qa-cycle-overlay}
export PYTHONPATH="$PWD:$CYCLE_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

if [[ ! -x "$STABILITY_PYTHON" ]]; then
  echo "Missing Python: $STABILITY_PYTHON" >&2
  exit 2
fi
# Reuse inherited HF_HOME/HF_HUB_CACHE, or Hugging Face's normal existing defaults.
# The one-command submitter checks cached snapshots before allocating a job.

"$STABILITY_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "No usable CUDA GPU allocated"'

EXTRA_ARGS=()
if [[ "${STABILITY_LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--local-files-only)
fi
if [[ -n "${STABILITY_REVISION:-}" ]]; then
  EXTRA_ARGS+=(--revision "$STABILITY_REVISION")
fi
if [[ -n "${STABILITY_LAYER:-}" ]]; then
  EXTRA_ARGS+=(--layer "$STABILITY_LAYER")
fi

"$STABILITY_PYTHON" -m qa_cycle_detector.stability_cli \
  --input "$STABILITY_INPUT" \
  --run-root "$STABILITY_RUN_ROOT" \
  --model "$STABILITY_MODEL" \
  --embedding-model "$STABILITY_EMBEDDING_MODEL" \
  --embedding-device cpu \
  --max-examples "${STABILITY_MAX_EXAMPLES:-20}" \
  --rank "${STABILITY_RANK:-3}" \
  --directions "${STABILITY_DIRECTIONS:-4}" \
  --amplitudes 0.005 0.02 0.05 \
  --baseline-samples 4 \
  --max-new-tokens "${STABILITY_MAX_NEW_TOKENS:-24}" \
  "${EXTRA_ARGS[@]}"
