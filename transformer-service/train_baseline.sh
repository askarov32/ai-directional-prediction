#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
SANDSTONE_DIR="${SANDSTONE_DIR:-/Users/temporary/unik/sandstone experiment ROD}"
DATASET_DIR="${DATASET_DIR:-$ROOT_DIR/transformer-service/artifacts/sandstone}"
DATASET_PATH="${DATASET_PATH:-$DATASET_DIR/pairs.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/transformer-service/artifacts/checkpoints/baseline}"
EPOCHS="${EPOCHS:-200}"
DEVICE="${DEVICE:-cpu}"
D_MODEL="${D_MODEL:-128}"
N_HEADS="${N_HEADS:-4}"
ENC_DEPTH="${ENC_DEPTH:-4}"
DEC_DEPTH="${DEC_DEPTH:-4}"
LR="${LR:-1e-3}"
N_TOKENS="${N_TOKENS:-1024}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ ! -f "$DATASET_PATH" ]; then
  echo "Dataset not found at $DATASET_PATH, running data prep..."
  PYTHONPATH="$ROOT_DIR/transformer-service/src" "$PYTHON_BIN" -m transformer_service.cli \
    --sandstone-dir "$SANDSTONE_DIR" \
    --output-dir "$DATASET_DIR" \
    --build-pairs
fi

PYTHONPATH="$ROOT_DIR/transformer-service/src" "$PYTHON_BIN" -m transformer_service.train \
  --dataset "$DATASET_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --device "$DEVICE" \
  --d-model "$D_MODEL" \
  --n-heads "$N_HEADS" \
  --enc-depth "$ENC_DEPTH" \
  --dec-depth "$DEC_DEPTH" \
  --learning-rate "$LR" \
  --n-tokens "$N_TOKENS"
