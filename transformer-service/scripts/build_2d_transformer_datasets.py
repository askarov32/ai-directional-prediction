from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from transformer_service.dataset import (
    INPUT_CHANNEL_NAMES,
    TARGET_CHANNEL_NAMES,
    SandstoneTensors,
    save_sandstone_artifacts,
)


DEFAULT_INPUT_ROOT = Path("pinn-service/artifacts/rod_experiments_2d")
DEFAULT_OUTPUT_ROOT = Path("transformer-service/artifacts/datasets_2d")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-convert strict 2D PINN structured datasets into per-rock Transformer pair datasets."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rocks", nargs="+", default=None, help="Optional subset of rocks to convert.")
    parser.add_argument(
        "--dataset-name-template",
        default="{rock}_transformer_2d",
        help="Output dataset directory name template. Available key: {rock}",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    datasets = discover_datasets(input_root, args.rocks)
    manifest_rows: list[dict[str, Any]] = []

    for rock, paths in datasets.items():
        payload = np.load(paths["structured_dataset"])
        tensors = build_tensors_from_structured(payload)
        dataset_dir = output_root / args.dataset_name_template.format(rock=rock)
        written = save_sandstone_artifacts(tensors, dataset_dir)
        enrich_metadata_file(
            metadata_path=written["metadata"],
            structured_path=paths["structured_dataset"],
            source_metadata_path=paths["metadata"],
            rock=rock,
        )
        manifest_rows.append(
            {
                "rock": rock,
                "source_structured_dataset": str(paths["structured_dataset"]),
                "source_metadata": str(paths["metadata"]),
                "output_dir": str(dataset_dir),
                "pairs_path": str(written["pairs"]),
                "scalers_path": str(written["scalers"]),
                "metadata_path": str(written["metadata"]),
                "node_count": int(tensors.state.shape[0]),
                "time_steps": int(tensors.state.shape[1]),
                "input_channels": int(tensors.state.shape[2]),
                "target_channels": len(tensors.target_channel_names),
            }
        )
        print(f"[{rock}] Transformer dataset: {dataset_dir}")

    manifest = {
        "source_root": str(input_root),
        "output_root": str(output_root),
        "dataset_count": len(manifest_rows),
        "datasets": manifest_rows,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {manifest_path}")


def discover_datasets(input_root: Path, selected_rocks: list[str] | None) -> dict[str, dict[str, Path]]:
    manifest_path = input_root / "manifest.json"
    selected = {value.strip().lower() for value in selected_rocks} if selected_rocks else None
    discovered: dict[str, dict[str, Path]] = {}

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for rock_info in manifest.get("rocks", []):
            rock = str(rock_info.get("rock", "")).strip().lower()
            if not rock:
                continue
            if selected is not None and rock not in selected:
                continue
            structured_dataset = Path(rock_info["structured_dataset"]).expanduser().resolve()
            metadata = Path(rock_info["metadata"]).expanduser().resolve()
            ensure_exists(structured_dataset, "structured dataset")
            ensure_exists(metadata, "dataset metadata")
            discovered[rock] = {
                "structured_dataset": structured_dataset,
                "metadata": metadata,
            }
        if discovered:
            return discovered

    for structured_dataset in sorted(input_root.glob("*/structured_dataset.npz")):
        rock = structured_dataset.parent.name.strip().lower()
        if selected is not None and rock not in selected:
            continue
        metadata = structured_dataset.parent / "dataset_metadata.json"
        ensure_exists(metadata, "dataset metadata")
        discovered[rock] = {
            "structured_dataset": structured_dataset.resolve(),
            "metadata": metadata.resolve(),
        }

    if not discovered:
        raise FileNotFoundError(
            f"No strict 2D PINN structured datasets found under {input_root}. "
            "Expected <rock>/structured_dataset.npz directories."
        )
    return discovered


def build_tensors_from_structured(payload: np.lib.npyio.NpzFile) -> SandstoneTensors:
    coords = payload["initial_coordinates"].astype(np.float32)
    times = payload["times"].astype(np.float32)
    temperature = payload["temperature"].astype(np.float32)
    displacement = payload["displacement"].astype(np.float32)
    velocity = payload["velocity"].astype(np.float32)
    material_static = payload["material_static"].astype(np.float32)
    thermal_properties = payload["thermal_properties"].astype(np.float32)

    n_nodes, n_times = temperature.shape
    coords_dynamic = np.broadcast_to(coords[:, None, :], (n_nodes, n_times, 3))
    materials_full = np.broadcast_to(
        material_static[:, None, :], (n_nodes, n_times, material_static.shape[1])
    )
    thermal_static_full = np.broadcast_to(
        np.stack([thermal_properties[:, 0, 0], thermal_properties[:, 0, 2]], axis=-1)[:, None, :],
        (n_nodes, n_times, 2),
    )

    state = np.concatenate(
        [
            coords_dynamic,
            temperature[..., None],
            displacement,
            velocity,
            materials_full,
            thermal_static_full,
        ],
        axis=-1,
    ).astype(np.float32)

    if state.shape[-1] != len(INPUT_CHANNEL_NAMES):
        raise ValueError(
            f"Structured dataset channel count mismatch: got {state.shape[-1]}, expected {len(INPUT_CHANNEL_NAMES)}"
        )

    target_indices = [INPUT_CHANNEL_NAMES.index(name) for name in TARGET_CHANNEL_NAMES]
    flat_in = state.reshape(-1, state.shape[-1])
    input_mean = flat_in.mean(axis=0)
    input_std = flat_in.std(axis=0)
    input_std = np.where(input_std < 1e-8, 1.0, input_std).astype(np.float32)
    target_mean = input_mean[target_indices].astype(np.float32)
    target_std = input_std[target_indices].astype(np.float32)

    return SandstoneTensors(
        state=state,
        coords=coords.astype(np.float32),
        times=times.astype(np.float32),
        input_channel_names=list(INPUT_CHANNEL_NAMES),
        target_channel_names=list(TARGET_CHANNEL_NAMES),
        input_mean=input_mean.astype(np.float32),
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
        raw_node_counts={
            "structured_dataset_nodes": int(coords.shape[0]),
            "structured_dataset_timesteps": int(times.shape[0]),
            "intersection": int(coords.shape[0]),
        },
    )


def enrich_metadata_file(
    *,
    metadata_path: Path,
    structured_path: Path,
    source_metadata_path: Path,
    rock: str,
) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "dataset_kind": "transformer_pairs_from_structured_2d",
            "rock_id": rock,
            "source_structured_dataset": str(structured_path),
            "source_metadata": str(source_metadata_path),
            "dimension": 2,
            "effective_domain_type": "rect_2d",
        }
    )
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


if __name__ == "__main__":
    main()
