#!/usr/bin/env bash
set -euo pipefail

mkdir -p slurm/logs slurm/state

HALOSCOPE_DIR=${HALOSCOPE_DIR:-$HOME/LLM_Haloscope}
PYTHON_BIN=${HALOSCOPE_PYTHON:-$HALOSCOPE_DIR/.venv/bin/python}
CYCLE_OVERLAY=${CYCLE_OVERLAY:-${SCRATCH:-$HOME}/qa-cycle-overlay}
RUN_ROOT=${CYCLE_RUN_ROOT:-${SCRATCH:-$HOME}/qa-cycle-results/llama2-triviaqa}
SOURCE_PREDICTIONS=${SOURCE_PREDICTIONS:?Set SOURCE_PREDICTIONS before submission}
RUN_NAME=$(basename "$RUN_ROOT")
STATE_FILE=${PIPELINE_STATE_FILE:-slurm/state/${RUN_NAME}.step}

export HALOSCOPE_DIR PYTHON_BIN CYCLE_OVERLAY RUN_ROOT SOURCE_PREDICTIONS
export PYTHONPATH="$PWD:$CYCLE_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing HaloScope Python: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -d "$CYCLE_OVERLAY" ]]; then
  echo "Missing dependency overlay: $CYCLE_OVERLAY" >&2
  exit 2
fi
if [[ -z "${HF_HOME:-}" ]]; then
  echo "Export the existing shared HF_HOME before submission" >&2
  exit 2
fi

"$PYTHON_BIN" -c "import torch, transformers, sentence_transformers, qa_cycle_detector; print('Python environment OK')"

if [[ "${RESET_PIPELINE:-0}" == "1" ]]; then
  rm -f "$STATE_FILE" "${STATE_FILE}.tmp" "${STATE_FILE}.done"
fi

ACCOUNT_ARGS=()
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
  ACCOUNT_ARGS=(--account "$SLURM_ACCOUNT")
fi

JOB_ID=$(sbatch --parsable --export=ALL,PIPELINE_STATE_FILE="$STATE_FILE" \
  --output=slurm/logs/%x-%j.out --error=slurm/logs/%x-%j.err \
  "${ACCOUNT_ARGS[@]}" --job-name=qa-cycle \
  slurm/cycle_pipeline.sbatch)

cat <<EOF
Submitted Q-A cycle pipeline: $JOB_ID
Results: $RUN_ROOT
State: $STATE_FILE
Monitor: squeue -j $JOB_ID
Log: slurm/logs/qa-cycle-$JOB_ID.out
EOF
