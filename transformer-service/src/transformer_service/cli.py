from __future__ import annotations

import argparse
from pathlib import Path

from transformer_service.dataset import build_sandstone_tensors, save_sandstone_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the transformer training dataset from raw COMSOL sandstone CSVs.",
    )
    parser.add_argument(
        "--sandstone-dir",
        type=Path,
        required=True,
        help="Directory containing data_materials.csv, data_temperature.csv, data_displacement.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where pairs.npz, scalers.json, dataset_metadata.json are written.",
    )
    parser.add_argument(
        "--build-pairs",
        action="store_true",
        help="Build the autoregressive pairs dataset (only mode currently supported).",
    )
    args = parser.parse_args()

    if not args.build_pairs:
        parser.error("Only --build-pairs mode is supported currently.")

    tensors = build_sandstone_tensors(args.sandstone_dir)
    paths = save_sandstone_artifacts(tensors, args.output_dir)
    print(
        "Built sandstone artifacts:\n"
        f"  pairs:    {paths['pairs']}\n"
        f"  scalers:  {paths['scalers']}\n"
        f"  metadata: {paths['metadata']}\n"
        f"  nodes (intersection): {tensors.state.shape[0]}\n"
        f"  time steps: {tensors.state.shape[1]}\n"
        f"  input channels: {tensors.state.shape[2]}\n"
        f"  target channels: {len(tensors.target_channel_names)}"
    )


if __name__ == "__main__":
    main()
