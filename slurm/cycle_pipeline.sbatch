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
RUN_ROOT=${CYCLE_RUN_ROOT:-${SCRATCH:-$HOME}/qa-cycle-results/llama2-triviaqa}
SOURCE_PREDICTIONS=${SOURCE_PREDICTIONS:?Set SOURCE_PREDICTIONS to PEFT TriviaQA predictions.jsonl}
MODEL_ID=${MODEL_ID:-meta-llama/Llama-2-7b-chat-hf}
EMBEDDING_MODEL=${EMBEDDING_MODEL:-sentence-transformers/all-mpnet-base-v2}
MAX_EXAMPLES=${MAX_EXAMPLES:-500}
SHARD_SIZE=${SHARD_SIZE:-10}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-0}
TOTAL_STEPS=7

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:$CYCLE_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "HaloScope Python is not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$SOURCE_PREDICTIONS" ]]; then
  echo "Missing source predictions: $SOURCE_PREDICTIONS" >&2
  exit 2
fi
if [[ -z "${HF_HOME:-}" ]]; then
  echo "HF_HOME must point to the existing shared HaloScope cache" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT" slurm/state
RUN_NAME=$(basename "$RUN_ROOT")
STATE_FILE=${PIPELINE_STATE_FILE:-slurm/state/${RUN_NAME}.step}

"$PYTHON_BIN" - <<'PY'
import torch
print("Job CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("The Slurm job did not receive a usable CUDA GPU")
print("Job GPU:", torch.cuda.get_device_name(0))
PY

requeue_current() {
  echo "Time-limit warning received; requeuing job $SLURM_JOB_ID at step ${STEP:-unknown}."
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_current USR1

if [[ -f "${STATE_FILE}.done" ]]; then
  echo "Pipeline already complete: ${STATE_FILE}.done"
  exit 0
fi
STEP=0
if [[ -f "$STATE_FILE" ]]; then
  STEP=$(<"$STATE_FILE")
fi
if ! [[ "$STEP" =~ ^[0-9]+$ ]] || (( STEP > TOTAL_STEPS )); then
  echo "Invalid state '$STEP' in $STATE_FILE" >&2
  exit 2
fi

GENERATION_ARGS=(
  --generator-model "$MODEL_ID"
  --embedding-model "$EMBEDDING_MODEL"
  --embedding-device cpu
  --torch-dtype float16
  --forward-samples 10
  --inverse-samples 10
  --forward-temperature 0.8
  --inverse-temperature 0.8
  --threshold 0.82
  --smoothing 0.5
  --seed 42
)
if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  GENERATION_ARGS+=(--local-files-only)
fi

while (( STEP < TOTAL_STEPS )); do
  case "$STEP" in
    0)
      LABEL="prepare-data"
      CMD=("$PYTHON_BIN" scripts/prepare_peft_triviaqa.py
        --input "$SOURCE_PREDICTIONS" --output-dir "$RUN_ROOT/data"
        --seed 42 --max-examples "$MAX_EXAMPLES")
      ;;
    1)
      LABEL="extract-train"
      CMD=("$PYTHON_BIN" -m qa_cycle_detector.cli extract-sharded
        --input "$RUN_ROOT/data/train.jsonl"
        --output-dir "$RUN_ROOT/features/train" --shard-size "$SHARD_SIZE"
        --keep-samples "${GENERATION_ARGS[@]}")
      ;;
    2)
      LABEL="extract-validation"
      CMD=("$PYTHON_BIN" -m qa_cycle_detector.cli extract-sharded
        --input "$RUN_ROOT/data/validation.jsonl"
        --output-dir "$RUN_ROOT/features/validation" --shard-size "$SHARD_SIZE"
        --keep-samples "${GENERATION_ARGS[@]}")
      ;;
    3)
      LABEL="train-ablations"
      CMD=("$PYTHON_BIN" scripts/run_ablations.py train
        --input "$RUN_ROOT/features/train" --output-dir "$RUN_ROOT/models")
      ;;
    4)
      LABEL="evaluate-validation"
      CMD=("$PYTHON_BIN" scripts/run_ablations.py evaluate
        --input "$RUN_ROOT/features/validation" --model-dir "$RUN_ROOT/models"
        --output "$RUN_ROOT/validation_metrics.json")
      ;;
    5)
      LABEL="extract-test"
      CMD=("$PYTHON_BIN" -m qa_cycle_detector.cli extract-sharded
        --input "$RUN_ROOT/data/test.jsonl"
        --output-dir "$RUN_ROOT/features/test" --shard-size "$SHARD_SIZE"
        --keep-samples "${GENERATION_ARGS[@]}")
      ;;
    6)
      LABEL="evaluate-test"
      CMD=("$PYTHON_BIN" scripts/run_ablations.py evaluate
        --input "$RUN_ROOT/features/test" --model-dir "$RUN_ROOT/models"
        --output "$RUN_ROOT/test_metrics.json")
      ;;
  esac

  echo "Starting step $STEP/$((TOTAL_STEPS - 1)): $LABEL"
  "${CMD[@]}" &
  CHILD_PID=$!
  wait "$CHILD_PID"
  unset CHILD_PID
  STEP=$((STEP + 1))
  printf '%s\n' "$STEP" > "${STATE_FILE}.tmp"
  mv "${STATE_FILE}.tmp" "$STATE_FILE"
done

touch "${STATE_FILE}.done"
echo "Pipeline complete. Results: $RUN_ROOT"
