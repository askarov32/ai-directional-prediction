#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv-pinn/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/fno-service/configs/train_fno.yaml}"
PINN_STRUCTURED_PATH="${PINN_STRUCTURED_PATH:-$ROOT_DIR/pinn-service/artifacts/demo/structured_dataset.npz}"
PINN_METADATA_PATH="${PINN_METADATA_PATH:-$ROOT_DIR/pinn-service/artifacts/demo/dataset_metadata.json}"
DATASET_DIR="${1:-${DATASET_DIR:-$ROOT_DIR/fno-service/artifacts/datasets/sandstone_fno}}"
OUTPUT_DIR="${2:-${OUTPUT_DIR:-$ROOT_DIR/fno-service/artifacts/checkpoints/baseline}}"
GRID_Z="${GRID_Z:-1}"
GRID_Y="${GRID_Y:-32}"
GRID_X="${GRID_X:-32}"
MAX_TIMESTEPS="${MAX_TIMESTEPS:-64}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.000001}"
WIDTH="${WIDTH:-16}"
MODES_X="${MODES_X:-6}"
MODES_Y="${MODES_Y:-6}"
DEPTH="${DEPTH:-2}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cpu}"

echo "Preparing FNO baseline dataset..."
PYTHONPATH="$ROOT_DIR/fno-service/src" "$PYTHON_BIN" "$ROOT_DIR/fno-service/scripts/prepare_fno_dataset.py" \
  --pinn-structured "$PINN_STRUCTURED_PATH" \
  --pinn-metadata "$PINN_METADATA_PATH" \
  --output-dir "$DATASET_DIR" \
  --grid-res "$GRID_Z" "$GRID_Y" "$GRID_X" \
  --max-timesteps "$MAX_TIMESTEPS" \
  --validate

TRAIN_ARGS=(
  --config "$CONFIG_PATH"
  --dataset-path "$DATASET_DIR"
  --output-dir "$OUTPUT_DIR"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY"
  --width "$WIDTH"
  --modes-x "$MODES_X"
  --modes-y "$MODES_Y"
  --depth "$DEPTH"
  --val-fraction "$VAL_FRACTION"
  --seed "$SEED"
  --device "$DEVICE"
)

if [[ -n "$SAMPLE_LIMIT" ]]; then
  TRAIN_ARGS+=(--sample-limit "$SAMPLE_LIMIT")
fi

echo "Training FNO baseline checkpoint..."
PYTHONPATH="$ROOT_DIR/fno-service/src" "$PYTHON_BIN" "$ROOT_DIR/fno-service/scripts/train_fno.py" \
  "${TRAIN_ARGS[@]}"
