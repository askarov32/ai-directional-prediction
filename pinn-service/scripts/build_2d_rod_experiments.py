from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from pinn_service.dataset_builder import (
    STRAIN_FIELDS,
    STRESS_NORMAL_FIELDS,
    STRESS_SHEAR_FIELDS,
    TRAINING_INPUT_NAMES,
    TRAINING_TARGET_NAMES,
)


ROCKS = ("granite", "limestone", "sandstone", "basalt")
PlanePolicy = Literal["max_nodes", "nearest_z"]
OutOfPlaneMode = Literal["zero", "preserve"]


@dataclass(frozen=True)
class Rock2DArtifacts:
    rock_id: str
    source_structured_path: Path
    source_metadata_path: Path
    structured_path: Path
    metadata_path: Path
    training_matrix_path: Path | None
    row_count: int
    node_count: int
    selected_plane_z: float
    selected_plane_node_count: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build strict 2D rod experiment artifacts from existing PINN structured 3D artifacts."
    )
    parser.add_argument(
        "--input-root",
        default="pinn-service/artifacts/rod_experiments",
        help="Root containing per-rock structured_dataset.npz folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="pinn-service/artifacts/rod_experiments_2d",
        help="Root where 2D per-rock artifacts and combined training matrix are written.",
    )
    parser.add_argument("--rocks", nargs="+", default=list(ROCKS), choices=ROCKS)
    parser.add_argument(
        "--plane-policy",
        choices=("max_nodes", "nearest_z"),
        default="max_nodes",
        help=(
            "max_nodes selects the z-plane with the most mesh nodes; nearest_z selects nodes nearest to --plane-z. "
            "Use nearest_z with --plane-z 0 for the central cross-section."
        ),
    )
    parser.add_argument("--plane-z", type=float, default=0.0, help="Target z-plane used by --plane-policy nearest_z.")
    parser.add_argument(
        "--z-round-decimals",
        type=int,
        default=8,
        help="Decimal precision used when grouping mesh nodes into z-planes.",
    )
    parser.add_argument(
        "--z-tolerance",
        type=float,
        default=None,
        help="Optional absolute tolerance around --plane-z. If omitted, nearest exact rounded z-plane is used.",
    )
    parser.add_argument(
        "--out-of-plane-mode",
        choices=("zero", "preserve"),
        default="zero",
        help="zero creates strict 2D targets with z coordinates, disp_z, and vel_z set to 0.",
    )
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument(
        "--skip-training-matrix",
        action="store_true",
        help="Only write structured_dataset.npz and dataset_metadata.json per rock.",
    )
    parser.add_argument(
        "--skip-combined",
        action="store_true",
        help="Do not concatenate per-rock 2D training matrices.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    artifacts: list[Rock2DArtifacts] = []
    for rock_id in args.rocks:
        artifact = build_rock_2d_artifacts(
            rock_id=rock_id,
            source_dir=input_root / rock_id,
            output_dir=output_root / rock_id,
            plane_policy=args.plane_policy,
            plane_z=args.plane_z,
            z_round_decimals=args.z_round_decimals,
            z_tolerance=args.z_tolerance,
            out_of_plane_mode=args.out_of_plane_mode,
            dtype=args.dtype,
            build_training_matrix=not args.skip_training_matrix,
        )
        artifacts.append(artifact)
        print(f"[{rock_id}] 2D structured dataset: {artifact.structured_path}")
        print(f"[{rock_id}] 2D metadata: {artifact.metadata_path}")
        if artifact.training_matrix_path is not None:
            print(f"[{rock_id}] 2D training matrix: {artifact.training_matrix_path}")
        print(f"[{rock_id}] selected z={artifact.selected_plane_z:g}, nodes={artifact.selected_plane_node_count}")

    combined_summary: dict[str, Any] | None = None
    if not args.skip_training_matrix and not args.skip_combined:
        combined_summary = write_combined_training_matrix(
            artifacts=artifacts,
            output_path=output_root / "training_samples_all_rocks.npz",
            metadata_path=output_root / "training_samples_all_rocks_metadata.json",
            out_of_plane_mode=args.out_of_plane_mode,
        )
        print("Combined 2D training matrix:", output_root / "training_samples_all_rocks.npz")
        print("Combined 2D metadata:", output_root / "training_samples_all_rocks_metadata.json")

    manifest_path = output_root / "manifest.json"
    write_manifest(
        manifest_path,
        input_root=input_root,
        output_root=output_root,
        artifacts=artifacts,
        args=args,
        combined_summary=combined_summary,
    )
    print("2D manifest:", manifest_path)


def build_rock_2d_artifacts(
    *,
    rock_id: str,
    source_dir: Path,
    output_dir: Path,
    plane_policy: PlanePolicy,
    plane_z: float,
    z_round_decimals: int,
    z_tolerance: float | None,
    out_of_plane_mode: OutOfPlaneMode,
    dtype: str,
    build_training_matrix: bool,
) -> Rock2DArtifacts:
    source_structured_path = source_dir / "structured_dataset.npz"
    source_metadata_path = source_dir / "dataset_metadata.json"
    if not source_structured_path.exists():
        raise FileNotFoundError(f"Missing source structured dataset: {source_structured_path}")
    if not source_metadata_path.exists():
        raise FileNotFoundError(f"Missing source metadata: {source_metadata_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    source = np.load(source_structured_path)
    source_coords = source["initial_coordinates"]
    selected = select_plane_indices(
        source_coords[:, 2],
        policy=plane_policy,
        plane_z=plane_z,
        decimals=z_round_decimals,
        tolerance=z_tolerance,
    )
    if selected.indices.size == 0:
        raise ValueError(f"No nodes selected for {rock_id} with plane policy {plane_policy}.")

    np_dtype = np.dtype(dtype)
    sliced = slice_structured_payload(
        source,
        selected.indices,
        dtype=np_dtype,
        out_of_plane_mode=out_of_plane_mode,
    )

    structured_path = output_dir / "structured_dataset.npz"
    np.savez_compressed(structured_path, **sliced)

    training_matrix_path: Path | None = None
    row_count = 0
    if build_training_matrix:
        training_matrix_path = output_dir / "training_samples.npz"
        inputs, targets = build_training_matrix_from_structured(sliced)
        row_count = int(inputs.shape[0])
        np.savez_compressed(
            training_matrix_path,
            inputs=inputs.astype(np_dtype, copy=False),
            targets=targets.astype(np_dtype, copy=False),
            input_feature_names=np.asarray(TRAINING_INPUT_NAMES, dtype="<U32"),
            target_feature_names=np.asarray(TRAINING_TARGET_NAMES, dtype="<U32"),
        )

    metadata_path = output_dir / "dataset_metadata.json"
    metadata = build_2d_metadata(
        source_metadata=source_metadata,
        rock_id=rock_id,
        source_structured_path=source_structured_path,
        source_metadata_path=source_metadata_path,
        structured_path=structured_path,
        training_matrix_path=training_matrix_path,
        selected=selected,
        selected_indices=selected.indices,
        sliced=sliced,
        plane_policy=plane_policy,
        plane_z=plane_z,
        z_tolerance=z_tolerance,
        out_of_plane_mode=out_of_plane_mode,
        row_count=row_count,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return Rock2DArtifacts(
        rock_id=rock_id,
        source_structured_path=source_structured_path,
        source_metadata_path=source_metadata_path,
        structured_path=structured_path,
        metadata_path=metadata_path,
        training_matrix_path=training_matrix_path,
        row_count=row_count,
        node_count=int(sliced["initial_coordinates"].shape[0]),
        selected_plane_z=float(selected.plane_z),
        selected_plane_node_count=int(selected.indices.shape[0]),
    )


@dataclass(frozen=True)
class PlaneSelection:
    plane_z: float
    indices: np.ndarray
    candidate_plane_counts: dict[str, int]


def select_plane_indices(
    z_values: np.ndarray,
    *,
    policy: PlanePolicy,
    plane_z: float,
    decimals: int,
    tolerance: float | None,
) -> PlaneSelection:
    rounded = np.round(z_values.astype(float), decimals=decimals)
    unique_z, counts = np.unique(rounded, return_counts=True)
    plane_counts = {f"{float(z):.12g}": int(count) for z, count in zip(unique_z, counts, strict=True)}

    if policy == "max_nodes":
        best_index = int(np.argmax(counts))
        selected_plane_z = float(unique_z[best_index])
        mask = rounded == selected_plane_z
    elif policy == "nearest_z":
        if tolerance is not None:
            mask = np.abs(z_values.astype(float) - plane_z) <= tolerance
            if not np.any(mask):
                raise ValueError(f"No nodes found within z tolerance {tolerance} around z={plane_z}.")
            selected_plane_z = float(np.mean(z_values[mask]))
        else:
            best_index = int(np.argmin(np.abs(unique_z - plane_z)))
            selected_plane_z = float(unique_z[best_index])
            mask = rounded == selected_plane_z
    else:
        raise ValueError(f"Unsupported plane policy: {policy}")

    indices = np.flatnonzero(mask)
    return PlaneSelection(
        plane_z=selected_plane_z,
        indices=indices.astype(np.int64),
        candidate_plane_counts=plane_counts,
    )


def slice_structured_payload(
    source: np.lib.npyio.NpzFile,
    node_indices: np.ndarray,
    *,
    dtype: np.dtype,
    out_of_plane_mode: OutOfPlaneMode,
) -> dict[str, np.ndarray]:
    sliced = {
        "times": source["times"].astype(dtype, copy=False),
        "initial_coordinates": source["initial_coordinates"][node_indices].astype(dtype, copy=True),
        "dynamic_coordinates": source["dynamic_coordinates"][node_indices].astype(dtype, copy=True),
        "material_static": source["material_static"][node_indices].astype(dtype, copy=False),
        "temperature": source["temperature"][node_indices].astype(dtype, copy=False),
        "thermal_properties": source["thermal_properties"][node_indices].astype(dtype, copy=False),
        "displacement": source["displacement"][node_indices].astype(dtype, copy=True),
        "velocity": source["velocity"][node_indices].astype(dtype, copy=True),
        "stress_normal": source["stress_normal"][node_indices].astype(dtype, copy=True),
        "stress_shear": source["stress_shear"][node_indices].astype(dtype, copy=True),
        "strain": source["strain"][node_indices].astype(dtype, copy=True),
    }
    if out_of_plane_mode == "zero":
        apply_strict_2d_zeroing(sliced)
    elif out_of_plane_mode != "preserve":
        raise ValueError(f"Unsupported out_of_plane_mode: {out_of_plane_mode}")
    return sliced


def apply_strict_2d_zeroing(payload: dict[str, np.ndarray]) -> None:
    payload["initial_coordinates"][:, 2] = 0.0
    payload["dynamic_coordinates"][:, :, 2] = 0.0
    payload["displacement"][:, :, 2] = 0.0
    payload["velocity"][:, :, 2] = 0.0
    payload["stress_normal"][:, :, 3] = 0.0
    payload["stress_shear"][:, :, 1:3] = 0.0
    payload["strain"][:, :, 2] = 0.0
    payload["strain"][:, :, 4:6] = 0.0


def build_training_matrix_from_structured(payload: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    times = payload["times"]
    initial_coordinates = payload["initial_coordinates"]
    material_static = payload["material_static"]
    thermal_properties = payload["thermal_properties"]
    temperature = payload["temperature"]
    displacement = payload["displacement"]
    velocity = payload["velocity"]
    stress_normal = payload["stress_normal"]
    stress_shear = payload["stress_shear"]
    strain = payload["strain"]
    node_count = initial_coordinates.shape[0]
    time_count = times.shape[0]

    base_coordinates = np.repeat(initial_coordinates[:, None, :], time_count, axis=1)
    tiled_times = np.broadcast_to(times[None, :, None], (node_count, time_count, 1))
    material_features = np.repeat(material_static[:, None, :], time_count, axis=1)
    thermal_static = thermal_properties[:, :, [0, 2]]
    inputs = np.concatenate([base_coordinates, tiled_times, material_features, thermal_static], axis=-1)
    targets = np.concatenate(
        [temperature[:, :, None], displacement, velocity, stress_normal, stress_shear, strain],
        axis=-1,
    )
    return inputs.reshape(node_count * time_count, inputs.shape[-1]), targets.reshape(node_count * time_count, targets.shape[-1])


def build_2d_metadata(
    *,
    source_metadata: dict[str, Any],
    rock_id: str,
    source_structured_path: Path,
    source_metadata_path: Path,
    structured_path: Path,
    training_matrix_path: Path | None,
    selected: PlaneSelection,
    selected_indices: np.ndarray,
    sliced: dict[str, np.ndarray],
    plane_policy: PlanePolicy,
    plane_z: float,
    z_tolerance: float | None,
    out_of_plane_mode: OutOfPlaneMode,
    row_count: int,
) -> dict[str, Any]:
    return {
        "rock_id": rock_id,
        "experiment_id": f"{rock_id}_rod_2d",
        "source_structured_dataset": str(source_structured_path),
        "source_metadata": str(source_metadata_path),
        "structured_dataset": str(structured_path),
        "training_matrix": str(training_matrix_path) if training_matrix_path else None,
        "derived_from_dimension": source_metadata.get("dimension"),
        "dimension": 2,
        "plane_policy": plane_policy,
        "requested_plane_z": plane_z,
        "selected_plane_z": selected.plane_z,
        "z_tolerance": z_tolerance,
        "out_of_plane_mode": out_of_plane_mode,
        "node_count": int(sliced["initial_coordinates"].shape[0]),
        "source_node_count": int(source_metadata.get("node_count", 0)),
        "time_steps": int(sliced["times"].shape[0]),
        "time_start": float(sliced["times"][0]),
        "time_end": float(sliced["times"][-1]),
        "time_step": float(sliced["times"][1] - sliced["times"][0]) if sliced["times"].shape[0] > 1 else 0.0,
        "training_rows": row_count,
        "selected_node_indices_summary": {
            "count": int(selected_indices.shape[0]),
            "min": int(selected_indices.min()) if selected_indices.size else None,
            "max": int(selected_indices.max()) if selected_indices.size else None,
        },
        "candidate_plane_counts": selected.candidate_plane_counts,
        "material_static_field_order": source_metadata.get("material_static_field_order", []),
        "thermal_properties_field_order": source_metadata.get("thermal_properties_field_order", []),
        "stress_normal_field_order": list(STRESS_NORMAL_FIELDS.keys()),
        "stress_shear_field_order": list(STRESS_SHEAR_FIELDS.keys()),
        "strain_field_order": list(STRAIN_FIELDS.keys()),
        "training_input_names": TRAINING_INPUT_NAMES,
        "training_target_names": TRAINING_TARGET_NAMES,
        "notes": [
            "This is a derived 2D artifact built from an existing structured 3D COMSOL/PINN dataset.",
            "The selected z-plane is projected to z=0 for strict rect_2d training and inference compatibility.",
            "out_of_plane_mode=zero sets disp_z, vel_z, stress_z, stress_yz, stress_xz, strain_z, strain_yz, and strain_xz to zero.",
            "The original 3D structured artifacts are left unchanged.",
        ],
    }


def write_combined_training_matrix(
    *,
    artifacts: list[Rock2DArtifacts],
    output_path: Path,
    metadata_path: Path,
    out_of_plane_mode: OutOfPlaneMode,
) -> dict[str, Any]:
    input_names: list[str] | None = None
    target_names: list[str] | None = None
    input_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    row_counts: dict[str, int] = {}

    for artifact in artifacts:
        if artifact.training_matrix_path is None:
            continue
        payload = np.load(artifact.training_matrix_path)
        current_input_names = payload["input_feature_names"].tolist()
        current_target_names = payload["target_feature_names"].tolist()
        if input_names is None:
            input_names = current_input_names
            target_names = current_target_names
        elif input_names != current_input_names or target_names != current_target_names:
            raise ValueError(f"Feature schema mismatch while combining {artifact.training_matrix_path}")
        inputs = payload["inputs"].astype(np.float32, copy=False)
        targets = payload["targets"].astype(np.float32, copy=False)
        input_chunks.append(inputs)
        target_chunks.append(targets)
        row_counts[artifact.rock_id] = int(inputs.shape[0])

    if not input_chunks or input_names is None or target_names is None:
        raise ValueError("No 2D training matrices were available to combine.")

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
        "dimension": 2,
        "out_of_plane_mode": out_of_plane_mode,
    }
    metadata_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_manifest(
    path: Path,
    *,
    input_root: Path,
    output_root: Path,
    artifacts: list[Rock2DArtifacts],
    args: argparse.Namespace,
    combined_summary: dict[str, Any] | None,
) -> None:
    payload = {
        "builder": "pinn-service/scripts/build_2d_rod_experiments.py",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "plane_policy": args.plane_policy,
        "plane_z": args.plane_z,
        "z_round_decimals": args.z_round_decimals,
        "z_tolerance": args.z_tolerance,
        "out_of_plane_mode": args.out_of_plane_mode,
        "dimension": 2,
        "experiments": [
            {
                "rock_id": artifact.rock_id,
                "source_structured_dataset": str(artifact.source_structured_path),
                "source_metadata": str(artifact.source_metadata_path),
                "structured_dataset": str(artifact.structured_path),
                "metadata": str(artifact.metadata_path),
                "training_matrix": str(artifact.training_matrix_path) if artifact.training_matrix_path else None,
                "node_count": artifact.node_count,
                "training_rows": artifact.row_count,
                "selected_plane_z": artifact.selected_plane_z,
                "selected_plane_node_count": artifact.selected_plane_node_count,
            }
            for artifact in artifacts
        ],
        "combined_training_matrix": str(output_root / "training_samples_all_rocks.npz") if combined_summary else None,
        "combined_training_matrix_metadata": (
            str(output_root / "training_samples_all_rocks_metadata.json") if combined_summary else None
        ),
        "combined_training_matrix_summary": combined_summary,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
