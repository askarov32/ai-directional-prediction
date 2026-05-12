from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from pinn_service.comsol_parser import ParsedComsolExport, parse_comsol_csv


MATERIAL_FIELDS = {
    "youngs_modulus": "solid.E (Pa)",
    "poissons_ratio": "solid.nu (1)",
    "density": "solid.rho (kg/m^3)",
    "thermal_expansion": "te1.alpha_iso (1/K)",
}

THERMAL_FIELDS = {
    "temperature_k": "T (K)",
    "coord_x_dynamic": "x (m)",
    "coord_y_dynamic": "y (m)",
    "coord_z_dynamic": "z (m)",
    "thermal_conductivity": "ht.k_iso (W/(m*K))",
    "thermal_density": "ht.rho (kg/m^3)",
    "heat_capacity": "ht.Cp (J/(kg*K))",
}

DISPLACEMENT_FIELDS = {
    "disp_x": "u (m)",
    "disp_y": "v (m)",
    "disp_z": "w (m)",
    "vel_x": "ut (m/s)",
    "vel_y": "vt (m/s)",
    "vel_z": "wt (m/s)",
}

STRESS_NORMAL_FIELDS = {
    "von_mises": "solid.mises (N/m^2)",
    "stress_x": "solid.sx (N/m^2)",
    "stress_y": "solid.sy (N/m^2)",
    "stress_z": "solid.sz (N/m^2)",
}

STRESS_SHEAR_FIELDS = {
    "stress_xy": "solid.sxy (N/m^2)",
    "stress_yz": "solid.syz (N/m^2)",
    "stress_xz": "solid.sxz (N/m^2)",
}

STRAIN_NORMAL_FIELDS = {
    "strain_x": "solid.eX (1)",
    "strain_y": "solid.eY (1)",
    "strain_z": "solid.eZ (1)",
}

STRAIN_SHEAR_FIELDS = {
    "strain_xy": "solid.eXY (1)",
    "strain_yz": "solid.eYZ (1)",
    "strain_xz": "solid.eXZ (1)",
}

STRAIN_FIELDS = {
    **STRAIN_NORMAL_FIELDS,
    **STRAIN_SHEAR_FIELDS,
}

TRAINING_INPUT_NAMES = [
    "x",
    "y",
    "z",
    "t",
    "youngs_modulus",
    "poissons_ratio",
    "density",
    "thermal_expansion",
    "thermal_conductivity",
    "heat_capacity",
]

TRAINING_TARGET_NAMES = [
    "temperature_k",
    "disp_x",
    "disp_y",
    "disp_z",
    "vel_x",
    "vel_y",
    "vel_z",
    "von_mises",
    "stress_x",
    "stress_y",
    "stress_z",
    "stress_xy",
    "stress_yz",
    "stress_xz",
    "strain_x",
    "strain_y",
    "strain_z",
    "strain_xy",
    "strain_yz",
    "strain_xz",
]


@dataclass(frozen=True)
class DatasetArtifacts:
    structured_path: Path
    metadata_path: Path
    training_matrix_path: Path | None


def build_dataset_from_exports(
    *,
    materials_path: str | Path,
    temperature_path: str | Path,
    displacement_path: str | Path,
    stress1_path: str | Path,
    stress2_path: str | Path,
    stress3_path: str | Path | None,
    output_dir: str | Path,
    strain_path: str | Path | None = None,
    mesh_path: str | Path | None = None,
    rock_id: str | None = None,
    experiment_id: str | None = None,
    coordinate_policy: Literal["strict", "intersection"] = "strict",
    dtype: str = "float32",
    build_training_matrix: bool = False,
) -> DatasetArtifacts:
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if strain_path is None and stress3_path is None:
        raise ValueError("Either strain_path or stress3_path must be provided.")

    exports: dict[str, ParsedComsolExport] = {
        "materials": parse_comsol_csv(materials_path),
        "temperature": parse_comsol_csv(temperature_path),
        "displacement": parse_comsol_csv(displacement_path),
        "stress1": parse_comsol_csv(stress1_path),
        "stress2": parse_comsol_csv(stress2_path),
    }
    if stress3_path is not None:
        exports["stress3"] = parse_comsol_csv(stress3_path)
    if strain_path is not None:
        exports["strain"] = parse_comsol_csv(strain_path)

    alignment = _resolve_coordinate_alignment(exports, coordinate_policy)
    np_dtype = np.dtype(dtype)

    canonical = exports["materials"]
    times = canonical.header.times.astype(np_dtype)
    initial_coordinates = _coordinates(exports["materials"], alignment.indices["materials"]).astype(np_dtype)

    material_static = np.stack(
        [_field(exports["materials"], field_name, alignment.indices["materials"])[:, 0] for field_name in MATERIAL_FIELDS.values()],
        axis=-1,
    ).astype(np_dtype)
    material_static_drift = _calculate_static_drift(exports["materials"], MATERIAL_FIELDS, alignment.indices["materials"])

    dynamic_coordinates = np.stack(
        [_field(exports["temperature"], field_name, alignment.indices["temperature"]) for field_name in ("x (m)", "y (m)", "z (m)")],
        axis=-1,
    ).astype(np_dtype)

    temperature = _field(exports["temperature"], THERMAL_FIELDS["temperature_k"], alignment.indices["temperature"]).astype(np_dtype)
    thermal_properties = np.stack(
        [
            _field(exports["temperature"], THERMAL_FIELDS["thermal_conductivity"], alignment.indices["temperature"]),
            _field(exports["temperature"], THERMAL_FIELDS["thermal_density"], alignment.indices["temperature"]),
            _field(exports["temperature"], THERMAL_FIELDS["heat_capacity"], alignment.indices["temperature"]),
        ],
        axis=-1,
    ).astype(np_dtype)

    displacement = np.stack(
        [
            _field(exports["displacement"], DISPLACEMENT_FIELDS["disp_x"], alignment.indices["displacement"]),
            _field(exports["displacement"], DISPLACEMENT_FIELDS["disp_y"], alignment.indices["displacement"]),
            _field(exports["displacement"], DISPLACEMENT_FIELDS["disp_z"], alignment.indices["displacement"]),
        ],
        axis=-1,
    ).astype(np_dtype)

    velocity = np.stack(
        [
            _field(exports["displacement"], DISPLACEMENT_FIELDS["vel_x"], alignment.indices["displacement"]),
            _field(exports["displacement"], DISPLACEMENT_FIELDS["vel_y"], alignment.indices["displacement"]),
            _field(exports["displacement"], DISPLACEMENT_FIELDS["vel_z"], alignment.indices["displacement"]),
        ],
        axis=-1,
    ).astype(np_dtype)

    stress_normal = np.stack(
        [_field(exports["stress1"], field_name, alignment.indices["stress1"]) for field_name in STRESS_NORMAL_FIELDS.values()],
        axis=-1,
    ).astype(np_dtype)

    stress_shear = np.stack(
        [_field(exports["stress2"], field_name, alignment.indices["stress2"]) for field_name in STRESS_SHEAR_FIELDS.values()],
        axis=-1,
    ).astype(np_dtype)

    strain, strain_source = _build_strain_tensor(exports, alignment.indices, np_dtype)

    reference_temperature_k = float(np.mean(temperature[:, 0]))

    structured_path = output_root / "structured_dataset.npz"
    np.savez_compressed(
        structured_path,
        times=times,
        initial_coordinates=initial_coordinates,
        dynamic_coordinates=dynamic_coordinates,
        material_static=material_static,
        temperature=temperature,
        thermal_properties=thermal_properties,
        displacement=displacement,
        velocity=velocity,
        stress_normal=stress_normal,
        stress_shear=stress_shear,
        strain=strain,
    )

    training_matrix_path: Path | None = None
    if build_training_matrix:
        training_matrix_path = output_root / "training_samples.npz"
        inputs, targets = _build_training_matrix(
            times=times,
            initial_coordinates=initial_coordinates,
            material_static=material_static,
            thermal_properties=thermal_properties,
            temperature=temperature,
            displacement=displacement,
            velocity=velocity,
            stress_normal=stress_normal,
            stress_shear=stress_shear,
            strain=strain,
        )
        np.savez_compressed(
            training_matrix_path,
            inputs=inputs,
            targets=targets,
            input_feature_names=np.asarray(TRAINING_INPUT_NAMES, dtype="<U32"),
            target_feature_names=np.asarray(TRAINING_TARGET_NAMES, dtype="<U32"),
        )

    metadata_path = output_root / "dataset_metadata.json"
    metadata = {
        "rock_id": rock_id,
        "experiment_id": experiment_id,
        "source_files": {name: str(export.path) for name, export in exports.items()},
        "mesh_path": str(Path(mesh_path).expanduser().resolve()) if mesh_path is not None else None,
        "dimension": int(canonical.header.dimension),
        "node_count": int(initial_coordinates.shape[0]),
        "raw_node_counts": alignment.raw_node_counts,
        "unique_coordinate_counts": alignment.unique_coordinate_counts,
        "dropped_node_counts": alignment.dropped_node_counts,
        "duplicate_coordinate_counts": alignment.duplicate_coordinate_counts,
        "coordinate_policy": coordinate_policy,
        "time_steps": int(len(times)),
        "time_start": float(times[0]),
        "time_end": float(times[-1]),
        "time_step": float(times[1] - times[0]) if len(times) > 1 else 0.0,
        "reference_temperature_k": reference_temperature_k,
        "material_static_field_order": list(MATERIAL_FIELDS.keys()),
        "thermal_properties_field_order": ["thermal_conductivity", "thermal_density", "heat_capacity"],
        "stress_normal_field_order": list(STRESS_NORMAL_FIELDS.keys()),
        "stress_shear_field_order": list(STRESS_SHEAR_FIELDS.keys()),
        "strain_field_order": list(STRAIN_FIELDS.keys()),
        "strain_source": strain_source,
        "training_input_names": TRAINING_INPUT_NAMES,
        "training_target_names": TRAINING_TARGET_NAMES,
        "material_static_drift_abs_max": material_static_drift,
        "notes": [
            "initial_coordinates are the raw node coordinates from the COMSOL exports",
            "dynamic_coordinates are taken from the temperature export x/y/z time-dependent columns",
            "material_static stores one value per node for each material field using the first time slice",
            "coordinate_policy=intersection aligns exports by common rounded coordinates and keeps the first row for duplicate coordinates",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return DatasetArtifacts(
        structured_path=structured_path,
        metadata_path=metadata_path,
        training_matrix_path=training_matrix_path,
    )


@dataclass(frozen=True)
class CoordinateAlignment:
    indices: dict[str, np.ndarray]
    raw_node_counts: dict[str, int]
    unique_coordinate_counts: dict[str, int]
    dropped_node_counts: dict[str, int]
    duplicate_coordinate_counts: dict[str, int]


def _resolve_coordinate_alignment(
    exports: dict[str, ParsedComsolExport],
    coordinate_policy: Literal["strict", "intersection"],
) -> CoordinateAlignment:
    _validate_time_and_dimension(exports)
    if coordinate_policy == "strict":
        _validate_strict_coordinates(exports)
        indices = {name: np.arange(export.coordinates.shape[0]) for name, export in exports.items()}
        raw_counts = {name: int(export.coordinates.shape[0]) for name, export in exports.items()}
        return CoordinateAlignment(
            indices=indices,
            raw_node_counts=raw_counts,
            unique_coordinate_counts=raw_counts,
            dropped_node_counts={name: 0 for name in exports},
            duplicate_coordinate_counts={name: 0 for name in exports},
        )
    if coordinate_policy != "intersection":
        raise ValueError(f"Unsupported coordinate_policy: {coordinate_policy}")

    coordinate_maps = {name: _first_index_by_coordinate(export.coordinates) for name, export in exports.items()}
    common_keys = set.intersection(*(set(mapping.keys()) for mapping in coordinate_maps.values()))
    if not common_keys:
        raise ValueError("No common coordinates found across COMSOL exports.")

    reference_name = "materials"
    reference_mapping = coordinate_maps[reference_name]
    ordered_keys = [key for key in reference_mapping if key in common_keys]
    indices = {
        name: np.asarray([mapping[key] for key in ordered_keys], dtype=np.int64)
        for name, mapping in coordinate_maps.items()
    }

    raw_counts = {name: int(export.coordinates.shape[0]) for name, export in exports.items()}
    unique_counts = {name: len(mapping) for name, mapping in coordinate_maps.items()}
    aligned_count = len(ordered_keys)
    return CoordinateAlignment(
        indices=indices,
        raw_node_counts=raw_counts,
        unique_coordinate_counts=unique_counts,
        dropped_node_counts={name: raw_counts[name] - aligned_count for name in exports},
        duplicate_coordinate_counts={name: raw_counts[name] - unique_counts[name] for name in exports},
    )


def _validate_time_and_dimension(exports: dict[str, ParsedComsolExport]) -> None:
    names = list(exports.keys())
    reference = exports[names[0]]
    reference_times = reference.header.times

    for name in names[1:]:
        current = exports[name]
        if current.header.dimension != reference.header.dimension:
            raise ValueError(
                f"Dimension mismatch: {reference.path.name} has {reference.header.dimension}, {current.path.name} has {current.header.dimension}"
            )
        if current.header.times.shape != reference_times.shape or not np.allclose(current.header.times, reference_times):
            raise ValueError(f"Time grid mismatch between {reference.path.name} and {current.path.name}")


def _validate_strict_coordinates(exports: dict[str, ParsedComsolExport]) -> None:
    names = list(exports.keys())
    reference = exports[names[0]]
    reference_coordinates = reference.coordinates
    reference_nodes = reference.coordinates.shape[0]

    for name in names[1:]:
        current = exports[name]
        if current.coordinates.shape[0] != reference_nodes:
            raise ValueError(
                f"Node count mismatch: {reference.path.name} has {reference_nodes}, {current.path.name} has {current.coordinates.shape[0]}. "
                "Use coordinate_policy='intersection' to align exports by common coordinates."
            )
        if current.coordinates.shape != reference_coordinates.shape or not np.allclose(current.coordinates, reference_coordinates):
            raise ValueError(
                f"Initial coordinate mismatch between {reference.path.name} and {current.path.name}. "
                "Use coordinate_policy='intersection' to align exports by common coordinates."
            )


def _first_index_by_coordinate(coordinates: np.ndarray, decimals: int = 12) -> dict[tuple[float, float, float], int]:
    mapping: dict[tuple[float, float, float], int] = {}
    rounded = np.round(coordinates, decimals=decimals)
    for index, row in enumerate(rounded):
        key = (float(row[0]), float(row[1]), float(row[2]))
        mapping.setdefault(key, index)
    return mapping


def _coordinates(export: ParsedComsolExport, indices: np.ndarray) -> np.ndarray:
    return export.coordinates[indices]


def _field(export: ParsedComsolExport, field_name: str, indices: np.ndarray) -> np.ndarray:
    return export.field(field_name)[indices]


def _build_strain_tensor(
    exports: dict[str, ParsedComsolExport],
    indices: dict[str, np.ndarray],
    dtype: np.dtype,
) -> tuple[np.ndarray, str]:
    if "strain" in exports:
        strain_export = exports["strain"]
        strain = np.stack(
            [_field(strain_export, field_name, indices["strain"]) for field_name in STRAIN_FIELDS.values()],
            axis=-1,
        ).astype(dtype)
        return strain, "data_strain.csv"

    stress3_export = exports["stress3"]
    normal_strain = np.stack(
        [_field(stress3_export, field_name, indices["stress3"]) for field_name in STRAIN_NORMAL_FIELDS.values()],
        axis=-1,
    )
    shear_placeholder = np.zeros((*normal_strain.shape[:2], len(STRAIN_SHEAR_FIELDS)), dtype=normal_strain.dtype)
    strain = np.concatenate([normal_strain, shear_placeholder], axis=-1).astype(dtype)
    return strain, "data_stress_3.csv_normals_with_zero_shear_placeholders"


def _calculate_static_drift(export: ParsedComsolExport, field_mapping: dict[str, str], indices: np.ndarray) -> dict[str, float]:
    drift: dict[str, float] = {}
    for logical_name, raw_name in field_mapping.items():
        values = _field(export, raw_name, indices)
        drift[logical_name] = float(np.max(np.abs(values - values[:, :1])))
    return drift


def _build_training_matrix(
    *,
    times: np.ndarray,
    initial_coordinates: np.ndarray,
    material_static: np.ndarray,
    thermal_properties: np.ndarray,
    temperature: np.ndarray,
    displacement: np.ndarray,
    velocity: np.ndarray,
    stress_normal: np.ndarray,
    stress_shear: np.ndarray,
    strain: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    node_count = initial_coordinates.shape[0]
    time_count = times.shape[0]

    base_coordinates = np.repeat(initial_coordinates[:, None, :], time_count, axis=1)
    tiled_times = np.broadcast_to(times[None, :, None], (node_count, time_count, 1))
    material_features = np.repeat(material_static[:, None, :], time_count, axis=1)
    thermal_static = thermal_properties[:, :, [0, 2]]

    # Fixed ordering for downstream PINN training.
    inputs = np.concatenate(
        [
            base_coordinates,
            tiled_times,
            material_features,
            thermal_static,
        ],
        axis=-1,
    )

    targets = np.concatenate(
        [
            temperature[:, :, None],
            displacement,
            velocity,
            stress_normal,
            stress_shear,
            strain,
        ],
        axis=-1,
    )

    return inputs.reshape(node_count * time_count, inputs.shape[-1]), targets.reshape(node_count * time_count, targets.shape[-1])
