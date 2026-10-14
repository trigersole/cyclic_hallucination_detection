#!/usr/bin/env bash
#SBATCH --partition=MGPU-TC2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=05:50:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --open-mode=append

set -euo pipefail

HALOSCOPE_DIR=${HALOSCOPE_DIR:-$HOME/LLM_Haloscope}
PYTHON_BIN=${HALOSCOPE_PYTHON:-$HALOSCOPE_DIR/.venv/bin/python}
CYCLE_OVERLAY=${CYCLE_OVERLAY:-${SCRATCH:-$HOME}/qa-cycle-overlay}
RUN_ROOT=${CYCLE_RUN_ROOT:?Set CYCLE_RUN_ROOT to the completed experiment}
EMBEDDING_MODEL=${EMBEDDING_MODEL:-sentence-transformers/all-mpnet-base-v2}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-0}

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:$CYCLE_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "HaloScope Python is not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -d "$RUN_ROOT/features/train" ]]; then
  echo "Missing completed feature run: $RUN_ROOT/features/train" >&2
  exit 2
fi

requeue_current() {
  echo "Time-limit warning received; requeuing comparison job $SLURM_JOB_ID."
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_current USR1

BACKFILL_ARGS=(
  --run-root "$RUN_ROOT"
  --embedding-model "$EMBEDDING_MODEL"
  --device cuda
)
if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  BACKFILL_ARGS+=(--local-files-only)
fi

"$PYTHON_BIN" -m scripts.backfill_interrogate "${BACKFILL_ARGS[@]}"

"$PYTHON_BIN" -m scripts.run_ablations train \
  --input "$RUN_ROOT/features_interrogate/train" \
  --output-dir "$RUN_ROOT/models_interrogate"

"$PYTHON_BIN" -m scripts.run_ablations evaluate \
  --input "$RUN_ROOT/features_interrogate/validation" \
  --model-dir "$RUN_ROOT/models_interrogate" \
  --output "$RUN_ROOT/validation_metrics_interrogate.json"

"$PYTHON_BIN" -m scripts.run_ablations evaluate \
  --input "$RUN_ROOT/features_interrogate/test" \
  --model-dir "$RUN_ROOT/models_interrogate" \
  --output "$RUN_ROOT/test_metrics_interrogate.json"

echo "InterrogateLLM comparison complete: $RUN_ROOT/test_metrics_interrogate.json"
