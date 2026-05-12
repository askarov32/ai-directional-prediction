#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
DATASET_PATH="${1:-$ROOT_DIR/pinn-service/artifacts/demo/training_samples.npz}"
OUTPUT_DIR="${2:-$ROOT_DIR/pinn-service/artifacts/checkpoints/baseline}"
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-120000}"
DEVICE="${DEVICE:-cpu}"
SUPERVISED_WEIGHT="${SUPERVISED_WEIGHT:-1.0}"
VELOCITY_WEIGHT="${VELOCITY_WEIGHT:-0.25}"
WAVE_RESIDUAL_WEIGHT="${WAVE_RESIDUAL_WEIGHT:-0.1}"
THERMAL_RESIDUAL_WEIGHT="${THERMAL_RESIDUAL_WEIGHT:-0.05}"
REFERENCE_TEMPERATURE_K="${REFERENCE_TEMPERATURE_K:-293.15}"
PHYSICS_MODE="${PHYSICS_MODE:-coupled_thermoelastic}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

PYTHONPATH="$ROOT_DIR/pinn-service/src" "$ROOT_DIR/.venv-pinn/bin/python" -m pinn_service.train \
  --dataset "$DATASET_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --sample-limit "$SAMPLE_LIMIT" \
  --supervised-weight "$SUPERVISED_WEIGHT" \
  --velocity-weight "$VELOCITY_WEIGHT" \
  --wave-residual-weight "$WAVE_RESIDUAL_WEIGHT" \
  --thermal-residual-weight "$THERMAL_RESIDUAL_WEIGHT" \
  --reference-temperature-k "$REFERENCE_TEMPERATURE_K" \
  --max-grad-norm "$MAX_GRAD_NORM" \
  --physics-mode "$PHYSICS_MODE"
