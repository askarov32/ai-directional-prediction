#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
DATASET_PATH="${1:-$ROOT_DIR/pinn-service/artifacts/demo/training_samples.npz}"
OUTPUT_DIR="${2:-$ROOT_DIR/pinn-service/artifacts/checkpoints/baseline}"
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-120000}"
DEVICE="${DEVICE:-cpu}"

PYTHONPATH="$ROOT_DIR/pinn-service/src" "$ROOT_DIR/.venv-pinn/bin/python" -m pinn_service.train \
  --dataset "$DATASET_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --sample-limit "$SAMPLE_LIMIT"
