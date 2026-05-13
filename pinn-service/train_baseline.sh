#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
DATASET_PATH="${1:-$ROOT_DIR/pinn-service/artifacts/demo/training_samples.npz}"
OUTPUT_DIR="${2:-$ROOT_DIR/pinn-service/artifacts/checkpoints/baseline}"
VAL_DATASET_PATH="${VAL_DATASET_PATH:-}"
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
VALIDATION_BATCH_SIZE="${VALIDATION_BATCH_SIZE:-}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-120000}"
VALIDATION_SAMPLE_LIMIT="${VALIDATION_SAMPLE_LIMIT:-}"
DEVICE="${DEVICE:-cpu}"
SUPERVISED_WEIGHT="${SUPERVISED_WEIGHT:-1.0}"
VELOCITY_WEIGHT="${VELOCITY_WEIGHT:-0.25}"
WAVE_RESIDUAL_WEIGHT="${WAVE_RESIDUAL_WEIGHT:-0.1}"
THERMAL_RESIDUAL_WEIGHT="${THERMAL_RESIDUAL_WEIGHT:-0.05}"
REFERENCE_TEMPERATURE_K="${REFERENCE_TEMPERATURE_K:-293.15}"
PHYSICS_MODE="${PHYSICS_MODE:-coupled_thermoelastic}"
LOSS_BALANCE_MODE="${LOSS_BALANCE_MODE:-fixed}"
LOSS_SCALE_REPORT="${LOSS_SCALE_REPORT:-}"
SUPERVISED_LOSS_SCALE="${SUPERVISED_LOSS_SCALE:-}"
VELOCITY_LOSS_SCALE="${VELOCITY_LOSS_SCALE:-}"
WAVE_RESIDUAL_LOSS_SCALE="${WAVE_RESIDUAL_LOSS_SCALE:-}"
THERMAL_RESIDUAL_LOSS_SCALE="${THERMAL_RESIDUAL_LOSS_SCALE:-}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

TRAIN_ARGS=(
  --dataset "$DATASET_PATH"
  --output-dir "$OUTPUT_DIR"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --device "$DEVICE"
  --sample-limit "$SAMPLE_LIMIT"
  --supervised-weight "$SUPERVISED_WEIGHT"
  --velocity-weight "$VELOCITY_WEIGHT"
  --wave-residual-weight "$WAVE_RESIDUAL_WEIGHT"
  --thermal-residual-weight "$THERMAL_RESIDUAL_WEIGHT"
  --reference-temperature-k "$REFERENCE_TEMPERATURE_K"
  --loss-balance-mode "$LOSS_BALANCE_MODE"
  --max-grad-norm "$MAX_GRAD_NORM"
  --physics-mode "$PHYSICS_MODE"
)

if [[ -n "$VAL_DATASET_PATH" ]]; then
  TRAIN_ARGS+=(--val-dataset "$VAL_DATASET_PATH")
fi
if [[ -n "$VALIDATION_BATCH_SIZE" ]]; then
  TRAIN_ARGS+=(--validation-batch-size "$VALIDATION_BATCH_SIZE")
fi
if [[ -n "$VALIDATION_SAMPLE_LIMIT" ]]; then
  TRAIN_ARGS+=(--validation-sample-limit "$VALIDATION_SAMPLE_LIMIT")
fi
if [[ -n "$LOSS_SCALE_REPORT" ]]; then
  TRAIN_ARGS+=(--loss-scale-report "$LOSS_SCALE_REPORT")
fi
if [[ -n "$SUPERVISED_LOSS_SCALE" ]]; then
  TRAIN_ARGS+=(--supervised-loss-scale "$SUPERVISED_LOSS_SCALE")
fi
if [[ -n "$VELOCITY_LOSS_SCALE" ]]; then
  TRAIN_ARGS+=(--velocity-loss-scale "$VELOCITY_LOSS_SCALE")
fi
if [[ -n "$WAVE_RESIDUAL_LOSS_SCALE" ]]; then
  TRAIN_ARGS+=(--wave-residual-loss-scale "$WAVE_RESIDUAL_LOSS_SCALE")
fi
if [[ -n "$THERMAL_RESIDUAL_LOSS_SCALE" ]]; then
  TRAIN_ARGS+=(--thermal-residual-loss-scale "$THERMAL_RESIDUAL_LOSS_SCALE")
fi

PYTHONPATH="$ROOT_DIR/pinn-service/src" "$ROOT_DIR/.venv-pinn/bin/python" -m pinn_service.train \
  "${TRAIN_ARGS[@]}"
