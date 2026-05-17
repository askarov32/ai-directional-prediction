from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fno_service.data.dataset import load_fno_grid_tensors
from fno_service.data.pinn_to_grid import convert_pinn_structured_to_fno_grid


DEFAULT_INPUT_ROOT = Path("pinn-service/artifacts/rod_experiments_2d")
DEFAULT_OUTPUT_ROOT = Path("fno-service/artifacts/datasets_2d")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-convert strict 2D PINN structured datasets into per-rock FNO datasets."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rocks", nargs="+", default=None, help="Optional subset of rocks to convert.")
    parser.add_argument("--grid-res", nargs=3, type=int, default=[1, 32, 32], metavar=("Z", "Y", "X"))
    parser.add_argument("--max-timesteps", type=int, default=64)
    parser.add_argument(
        "--dataset-name-template",
        default="{rock}_fno_2d",
        help="Output dataset directory name template. Available key: {rock}",
    )
    parser.add_argument("--validate", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    datasets = discover_datasets(input_root, args.rocks)
    manifest_rows: list[dict[str, Any]] = []

    for rock, paths in datasets.items():
        dataset_dir = output_root / args.dataset_name_template.format(rock=rock)
        metadata = convert_pinn_structured_to_fno_grid(
            structured_path=paths["structured_dataset"],
            metadata_path=paths["metadata"],
            output_dir=dataset_dir,
            grid_resolution=tuple(args.grid_res),
            max_timesteps=args.max_timesteps,
        )

        row: dict[str, Any] = {
            "rock": rock,
            "source_structured_dataset": str(paths["structured_dataset"]),
            "source_metadata": str(paths["metadata"]),
            "output_dir": str(dataset_dir),
            "grid_resolution_zyx": list(args.grid_res),
            "selected_timesteps": int(metadata.get("selected_timesteps", 0)),
            "full_timesteps": int(metadata.get("full_timesteps", 0)),
        }

        if args.validate:
            tensors = load_fno_grid_tensors(dataset_dir)
            row["validation"] = {
                "grid_dynamic_shape": list(tensors.grid_dynamic.shape),
                "grid_static_shape": list(tensors.grid_static.shape),
                "grid_masks_shape": list(tensors.grid_masks.shape),
                "grid_coords_shape": list(tensors.grid_coords.shape),
                "field_names": list(tensors.field_names),
                "static_feature_names": list(tensors.static_feature_names),
                "mask_names": list(tensors.mask_names),
            }

        manifest_rows.append(row)
        print(f"[{rock}] FNO dataset: {dataset_dir}")

    manifest = {
        "source_root": str(input_root),
        "output_root": str(output_root),
        "dataset_count": len(manifest_rows),
        "grid_resolution_zyx": list(args.grid_res),
        "max_timesteps": args.max_timesteps,
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


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


if __name__ == "__main__":
    main()
