#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

required_paths=(
  "fno-service/artifacts/checkpoints/baseline/best_model.pth"
  "fno-service/artifacts/datasets/sandstone_fno/grid_dynamic.npy"
  "pinn-service/artifacts/checkpoints/baseline_cpu/best_model.pth"
  "transformer-service/artifacts/checkpoints/smoke/best_model.pth"
  "mgn-service/datasets/sandstone_comsol_real/processed/graph.pt"
  "mgn-service/datasets/sandstone_comsol_real/processed/trajectories.pt"
  "mgn-service/outputs/checkpoints/best_model.pt"
)

missing=0
for path in "${required_paths[@]}"; do
  if [[ ! -f "$path" ]]; then
    printf 'missing: %s\n' "$path" >&2
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  printf '\nProduction assets are incomplete. Sync the missing runtime files before deploy.\n' >&2
  exit 1
fi

mkdir -p mgn-service/outputs/predictions
chmod a+rwx mgn-service/outputs/predictions

printf 'Production runtime assets are present.\n'
