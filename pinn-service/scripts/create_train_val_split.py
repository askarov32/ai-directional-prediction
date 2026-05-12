from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create deterministic train/validation splits for PINN training samples.")
    parser.add_argument(
        "--dataset",
        default="pinn-service/artifacts/rod_experiments/training_samples_all_rocks.npz",
        help="Combined training_samples_all_rocks.npz.",
    )
    parser.add_argument(
        "--metadata",
        default="pinn-service/artifacts/rod_experiments/training_samples_all_rocks_metadata.json",
        help="Combined training matrix metadata with per-rock row counts.",
    )
    parser.add_argument(
        "--output-dir",
        default="pinn-service/artifacts/rod_experiments/splits",
        help="Directory where train_samples.npz, val_samples.npz, and split_metadata.json are written.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_path = Path(args.dataset).expanduser().resolve()
    metadata_path = Path(args.metadata).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1.")

    split = create_split(
        dataset_path=dataset_path,
        metadata_path=metadata_path,
        output_dir=output_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    print("Train samples:", split["train_path"])
    print("Validation samples:", split["val_path"])
    print("Split metadata:", split["metadata_path"])


def create_split(
    *,
    dataset_path: Path,
    metadata_path: Path,
    output_dir: Path,
    val_fraction: float,
    seed: int,
) -> dict[str, object]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload = np.load(dataset_path)
    inputs = payload["inputs"]
    targets = payload["targets"]
    input_feature_names = payload["input_feature_names"]
    target_feature_names = payload["target_feature_names"]

    rng = np.random.default_rng(seed)
    train_indices: list[np.ndarray] = []
    val_indices: list[np.ndarray] = []
    ranges: dict[str, dict[str, int]] = {}

    start = 0
    for rock_id, row_count in metadata["row_counts"].items():
        stop = start + int(row_count)
        local_indices = np.arange(start, stop)
        rng.shuffle(local_indices)
        val_count = max(1, int(round(row_count * val_fraction)))
        val_indices.append(np.sort(local_indices[:val_count]))
        train_indices.append(np.sort(local_indices[val_count:]))
        ranges[rock_id] = {
            "start": start,
            "stop": stop,
            "rows": int(row_count),
            "train_rows": int(row_count - val_count),
            "val_rows": int(val_count),
        }
        start = stop

    train_index = np.concatenate(train_indices)
    val_index = np.concatenate(val_indices)

    train_path = output_dir / "train_samples.npz"
    val_path = output_dir / "val_samples.npz"
    split_metadata_path = output_dir / "split_metadata.json"

    np.savez_compressed(
        train_path,
        inputs=inputs[train_index],
        targets=targets[train_index],
        input_feature_names=input_feature_names,
        target_feature_names=target_feature_names,
    )
    np.savez_compressed(
        val_path,
        inputs=inputs[val_index],
        targets=targets[val_index],
        input_feature_names=input_feature_names,
        target_feature_names=target_feature_names,
    )

    split_metadata = {
        "source_dataset": str(dataset_path),
        "source_metadata": str(metadata_path),
        "seed": seed,
        "val_fraction": val_fraction,
        "train_path": str(train_path),
        "val_path": str(val_path),
        "train_rows": int(train_index.shape[0]),
        "val_rows": int(val_index.shape[0]),
        "total_rows": int(inputs.shape[0]),
        "per_rock": ranges,
    }
    split_metadata_path.write_text(json.dumps(split_metadata, indent=2), encoding="utf-8")

    return {
        "train_path": str(train_path),
        "val_path": str(val_path),
        "metadata_path": str(split_metadata_path),
        "train_rows": int(train_index.shape[0]),
        "val_rows": int(val_index.shape[0]),
    }


if __name__ == "__main__":
    main()
