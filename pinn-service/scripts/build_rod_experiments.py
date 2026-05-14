from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pinn_service.dataset_builder import DatasetArtifacts, build_dataset_from_exports


ROD_EXPERIMENT_FOLDERS = {
    "granite": ["granite", "granite experiment rod"],
    "limestone": ["limestone", "limestone experiment rod"],
    "sandstone": ["sandstone", "sandstone experiment ROD"],
    "basalt": ["basalt", "basalt experiment rod"],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build PINN-ready rod experiment datasets for multiple rocks.")
    parser.add_argument(
        "--raw-root",
        default="data",
        help="Directory containing the raw rock experiment folders, for example ./data/granite or older exports under ~/Downloads.",
    )
    parser.add_argument(
        "--output-dir",
        default="pinn-service/artifacts/rod_experiments",
        help="Directory where processed per-rock artifacts and manifest.json are written.",
    )
    parser.add_argument(
        "--rocks",
        nargs="+",
        default=list(ROD_EXPERIMENT_FOLDERS),
        choices=tuple(ROD_EXPERIMENT_FOLDERS),
        help="Rock experiments to build.",
    )
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument(
        "--skip-training-matrix",
        action="store_true",
        help="Only write structured_dataset.npz and dataset_metadata.json.",
    )
    parser.add_argument(
        "--skip-combined",
        action="store_true",
        help="Do not concatenate per-rock training matrices into training_samples_all_rocks.npz.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_root = Path(args.raw_root).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "coordinate_policy": "intersection",
        "experiments": [],
    }

    built_experiments: list[dict[str, object]] = []

    for rock_id in args.rocks:
        raw_dir = _resolve_raw_dir(raw_root, rock_id)
        artifacts = _build_rock(
            rock_id=rock_id,
            raw_dir=raw_dir,
            output_dir=output_root / rock_id,
            dtype=args.dtype,
            build_training_matrix=not args.skip_training_matrix,
        )
        experiment_entry = {
            "rock_id": rock_id,
            "raw_dir": str(raw_dir),
            "structured_dataset": str(artifacts.structured_path),
            "metadata": str(artifacts.metadata_path),
            "training_matrix": str(artifacts.training_matrix_path) if artifacts.training_matrix_path else None,
        }
        built_experiments.append(experiment_entry)
        manifest["experiments"].append(experiment_entry)
        print(f"[{rock_id}] Structured dataset: {artifacts.structured_path}")
        print(f"[{rock_id}] Metadata: {artifacts.metadata_path}")
        if artifacts.training_matrix_path is not None:
            print(f"[{rock_id}] Training matrix: {artifacts.training_matrix_path}")

    if not args.skip_training_matrix and not args.skip_combined:
        combined_path = output_root / "training_samples_all_rocks.npz"
        combined_metadata_path = output_root / "training_samples_all_rocks_metadata.json"
        combined_summary = _write_combined_training_matrix(
            experiments=built_experiments,
            output_path=combined_path,
            metadata_path=combined_metadata_path,
        )
        manifest["combined_training_matrix"] = str(combined_path)
        manifest["combined_training_matrix_metadata"] = str(combined_metadata_path)
        manifest["combined_training_matrix_summary"] = combined_summary
        print("Combined training matrix:", combined_path)
        print("Combined metadata:", combined_metadata_path)

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Manifest:", manifest_path)


def _resolve_raw_dir(raw_root: Path, rock_id: str) -> Path:
    candidates = [raw_root / folder_name for folder_name in ROD_EXPERIMENT_FOLDERS[rock_id]]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    expected = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Raw experiment directory for '{rock_id}' was not found. Checked: {expected}")


def _build_rock(
    *,
    rock_id: str,
    raw_dir: Path,
    output_dir: Path,
    dtype: str,
    build_training_matrix: bool,
) -> DatasetArtifacts:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw experiment directory does not exist: {raw_dir}")

    strain_path = raw_dir / "data_strain.csv"
    mesh_path = _find_mesh_path(raw_dir, rock_id)

    return build_dataset_from_exports(
        materials_path=raw_dir / "data_materials.csv",
        temperature_path=raw_dir / "data_temperature.csv",
        displacement_path=raw_dir / "data_displacement.csv",
        stress1_path=raw_dir / "data_stress_1.csv",
        stress2_path=raw_dir / "data_stress_2.csv",
        stress3_path=raw_dir / "data_stress_3.csv",
        strain_path=strain_path if strain_path.exists() else None,
        mesh_path=mesh_path,
        rock_id=rock_id,
        experiment_id=f"{rock_id}_rod",
        coordinate_policy="intersection",
        output_dir=output_dir,
        dtype=dtype,
        build_training_matrix=build_training_matrix,
    )


def _find_mesh_path(raw_dir: Path, rock_id: str) -> Path | None:
    candidates = [
        raw_dir / f"{rock_id}_mesh.csv",
        raw_dir / f"{rock_id}.mphtxt",
        raw_dir / f"{rock_id}_mesh.mphtxt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _write_combined_training_matrix(
    *,
    experiments: list[dict[str, object]],
    output_path: Path,
    metadata_path: Path,
) -> dict[str, object]:
    input_names: list[str] | None = None
    target_names: list[str] | None = None
    input_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    row_counts: dict[str, int] = {}

    for experiment in experiments:
        rock_id = str(experiment["rock_id"])
        matrix_value = experiment.get("training_matrix")
        if matrix_value is None:
            continue
        matrix_path = Path(str(matrix_value))
        payload = np.load(matrix_path)
        current_input_names = payload["input_feature_names"].tolist()
        current_target_names = payload["target_feature_names"].tolist()
        if input_names is None:
            input_names = current_input_names
            target_names = current_target_names
        elif current_input_names != input_names or current_target_names != target_names:
            raise ValueError(f"Feature schema mismatch while combining {matrix_path}")

        inputs = payload["inputs"].astype(np.float32, copy=False)
        targets = payload["targets"].astype(np.float32, copy=False)
        input_chunks.append(inputs)
        target_chunks.append(targets)
        row_counts[rock_id] = int(inputs.shape[0])

    if not input_chunks or input_names is None or target_names is None:
        raise ValueError("No training matrices were available to combine.")

    combined_inputs = np.concatenate(input_chunks, axis=0)
    combined_targets = np.concatenate(target_chunks, axis=0)
    np.savez_compressed(
        output_path,
        inputs=combined_inputs,
        targets=combined_targets,
        input_feature_names=np.asarray(input_names, dtype="<U32"),
        target_feature_names=np.asarray(target_names, dtype="<U32"),
    )

    summary = {
        "row_counts": row_counts,
        "total_rows": int(combined_inputs.shape[0]),
        "input_shape": list(combined_inputs.shape),
        "target_shape": list(combined_targets.shape),
        "input_feature_names": input_names,
        "target_feature_names": target_names,
    }
    metadata_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
