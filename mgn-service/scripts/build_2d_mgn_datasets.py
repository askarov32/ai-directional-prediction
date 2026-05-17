from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.graph_builder import build_graph_from_mesh
from src.data.normalizer import FeatureNormalizer
from src.data.pipeline import build_node_static
from src.data.scenario import default_scenario, normalize_scenario_schema, save_yaml


DEFAULT_INPUT_ROOT = Path("pinn-service/artifacts/rod_experiments_2d")
DEFAULT_REGISTRY_ROOT = Path("mgn-service/datasets")

DYNAMIC_FIELD_NAMES = [
    "solid.ex", "solid.exy", "solid.exz",
    "solid.ey", "solid.eyz", "solid.ez",
    "solid.mises",
    "solid.sx", "solid.sxy", "solid.sxz",
    "solid.sy", "solid.syz", "solid.sz",
    "t",
    "u", "ut", "v", "vt", "w", "wt",
]
DYNAMIC_FIELD_UNITS = {name: "?" for name in DYNAMIC_FIELD_NAMES}
DYNAMIC_FIELD_UNITS.update({
    "t": "K",
    "u": "m", "v": "m", "w": "m",
    "ut": "m/s", "vt": "m/s", "wt": "m/s",
})
MATERIAL_FEATURE_NAMES = [
    "young_modulus",
    "poisson_ratio",
    "density",
    "thermal_expansion",
    "thermal_conductivity",
    "heat_capacity",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build strict 2D MeshGraphNet datasets from rod_experiments_2d structured artifacts."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--registry-root", type=Path, default=DEFAULT_REGISTRY_ROOT)
    parser.add_argument("--rocks", nargs="+", default=None, help="Optional subset of rocks to convert.")
    parser.add_argument(
        "--dataset-id-template",
        default="{rock}_rod_2d",
        help="Target dataset id template. Available key: {rock}",
    )
    parser.add_argument("--k-nearest", type=int, default=12)
    parser.add_argument("--target-mode", choices=("delta", "absolute"), default="delta")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--min-std", type=float, default=1e-8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_root = args.input_root.expanduser().resolve()
    registry_root = args.registry_root.expanduser().resolve()
    registry_root.mkdir(parents=True, exist_ok=True)

    datasets = discover_datasets(input_root, args.rocks)
    manifest_rows: list[dict[str, Any]] = []

    for rock, paths in datasets.items():
        dataset_id = args.dataset_id_template.format(rock=rock)
        dataset_root = registry_root / dataset_id
        processed_dir = dataset_root / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        payload = np.load(paths["structured_dataset"])
        scenario = build_scenario(rock=rock, payload=payload)
        scenario["dataset_id"] = dataset_id
        scenario.setdefault("training", {})
        scenario["training"]["target_mode"] = args.target_mode
        scenario["training"]["train_ratio"] = float(args.train_ratio)
        scenario["training"]["val_ratio"] = float(args.val_ratio)
        scenario["training"]["test_ratio"] = max(0.0, 1.0 - float(args.train_ratio) - float(args.val_ratio))
        save_yaml(scenario, dataset_root / "scenario.yaml")

        graph, data, metadata, normalization = build_processed_dataset(
            payload=payload,
            dataset_id=dataset_id,
            rock=rock,
            k_nearest=args.k_nearest,
            target_mode=args.target_mode,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            min_std=args.min_std,
            scenario=scenario,
            source_structured_dataset=paths["structured_dataset"],
            source_metadata_path=paths["metadata"],
        )

        torch.save(graph, processed_dir / "graph.pt")
        torch.save(data, processed_dir / "trajectories.pt")
        (processed_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (processed_dir / "normalization.json").write_text(json.dumps(normalization, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (processed_dir / "dynamic_normalization.json").write_text(
            json.dumps(normalization["dynamic"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (processed_dir / "static_normalization.json").write_text(
            json.dumps(normalization["static"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_preview_csv(processed_dir / "preview.csv", metadata["field_names"], metadata["field_units"], payload)

        manifest_rows.append(
            {
                "rock": rock,
                "dataset_id": dataset_id,
                "dataset_root": str(dataset_root),
                "processed_dir": str(processed_dir),
                "source_structured_dataset": str(paths["structured_dataset"]),
                "source_metadata": str(paths["metadata"]),
                "node_count": metadata["n_nodes"],
                "time_steps": metadata["n_timesteps"],
                "field_names": metadata["field_names"],
                "node_in_dim": metadata["node_in_dim"],
                "edge_in_dim": metadata["edge_in_dim"],
            }
        )
        print(f"[{rock}] MGN dataset: {dataset_root}")

    manifest = {
        "source_root": str(input_root),
        "registry_root": str(registry_root),
        "dataset_count": len(manifest_rows),
        "datasets": manifest_rows,
    }
    manifest_path = registry_root / "manifest_2d.json"
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


def build_scenario(*, rock: str, payload: np.lib.npyio.NpzFile) -> dict[str, Any]:
    coords = payload["initial_coordinates"].astype(np.float32)
    times = payload["times"].astype(np.float32)
    temperature = payload["temperature"].astype(np.float32)
    material_static = payload["material_static"].astype(np.float32)
    thermal_properties = payload["thermal_properties"].astype(np.float32)

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = coords.mean(axis=0)
    xy_span = np.maximum(maxs[:2] - mins[:2], 1e-6)
    radius = float(max(min(float(np.min(xy_span)) * 0.12, 0.25), 1e-4))
    initial_temperature = float(np.max(temperature[:, 0]))
    background_temperature = float(np.min(temperature[:, 0]))
    mean_material = material_static.mean(axis=0)
    mean_thermal = thermal_properties[:, 0, :].mean(axis=0)

    scenario = default_scenario(f"{rock}_rod_2d")
    scenario["rock_type"] = rock
    scenario["geometry"]["dimension"] = 2
    scenario["geometry"]["graph_source"] = "knn"
    scenario["scenario"]["type"] = "heated_rod"
    scenario["scenario"]["source_center"] = [float(center[0]), float(center[1]), 0.0]
    scenario["scenario"]["source_radius"] = radius
    scenario["scenario"]["initial_temperature"] = initial_temperature
    scenario["scenario"]["background_temperature"] = background_temperature
    scenario["material"]["young_modulus"] = float(mean_material[0])
    scenario["material"]["poisson_ratio"] = float(mean_material[1])
    scenario["material"]["density"] = float(mean_material[2])
    scenario["material"]["thermal_expansion"] = float(mean_material[3])
    scenario["material"]["thermal_conductivity"] = float(mean_thermal[0])
    scenario["material"]["heat_capacity"] = float(mean_thermal[2])
    scenario["time"]["start"] = float(times[0]) if len(times) else 0.0
    scenario["time"]["end"] = float(times[-1]) if len(times) else 0.0
    scenario["time"]["step"] = float(times[1] - times[0]) if len(times) > 1 else 0.0
    scenario["paths"] = {"raw_dir": ""}
    return normalize_scenario_schema(scenario, dataset_id=f"{rock}_rod_2d")


def build_processed_dataset(
    *,
    payload: np.lib.npyio.NpzFile,
    dataset_id: str,
    rock: str,
    k_nearest: int,
    target_mode: str,
    train_ratio: float,
    val_ratio: float,
    min_std: float,
    scenario: dict[str, Any],
    source_structured_dataset: Path,
    source_metadata_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    coords = payload["initial_coordinates"].astype(np.float32)
    times = payload["times"].astype(np.float32)
    temperature = payload["temperature"].astype(np.float32).T  # (T, N)
    displacement = np.transpose(payload["displacement"].astype(np.float32), (1, 0, 2))  # (T,N,3)
    velocity = np.transpose(payload["velocity"].astype(np.float32), (1, 0, 2))  # (T,N,3)
    stress_normal = np.transpose(
        payload["stress_normal"].astype(np.float32), (1, 0, 2)
    )  # (T,N,4) — (xx, yy, zz, mises)
    stress_shear = np.transpose(
        payload["stress_shear"].astype(np.float32), (1, 0, 2)
    )  # (T,N,3) — (xy, xz, yz)
    strain = np.transpose(
        payload["strain"].astype(np.float32), (1, 0, 2)
    )  # (T,N,6) — (xx, yy, zz, xy, xz, yz)
    material_static = payload["material_static"].astype(np.float32)
    thermal_properties = payload["thermal_properties"].astype(np.float32)

    # Assemble 20-field dynamic state in the exact order the
    # sandstone_comsol_real checkpoint was trained on (its metadata
    # carries field_names; this dataset must match cardinality and
    # order to load state_dict cleanly).
    T_steps, N = temperature.shape
    dynamic_state = np.zeros((T_steps, N, len(DYNAMIC_FIELD_NAMES)), dtype=np.float32)
    dynamic_state[..., 0] = strain[..., 0]   # solid.ex (εxx)
    dynamic_state[..., 1] = strain[..., 3]   # solid.exy
    dynamic_state[..., 2] = strain[..., 4]   # solid.exz  — ~0 in plane-strain
    dynamic_state[..., 3] = strain[..., 1]   # solid.ey (εyy)
    dynamic_state[..., 4] = strain[..., 5]   # solid.eyz  — ~0 in plane-strain
    dynamic_state[..., 5] = strain[..., 2]   # solid.ez (εzz) — ~0 in plane-strain
    dynamic_state[..., 6] = stress_normal[..., 3]  # solid.mises
    dynamic_state[..., 7] = stress_normal[..., 0]  # solid.sx
    dynamic_state[..., 8] = stress_shear[..., 0]   # solid.sxy
    dynamic_state[..., 9] = stress_shear[..., 1]   # solid.sxz — ~0 in plane-strain
    dynamic_state[..., 10] = stress_normal[..., 1]  # solid.sy
    dynamic_state[..., 11] = stress_shear[..., 2]   # solid.syz — ~0 in plane-strain
    dynamic_state[..., 12] = stress_normal[..., 2]  # solid.sz  — nonzero in plane-strain
    dynamic_state[..., 13] = temperature  # t
    dynamic_state[..., 14] = displacement[..., 0]  # u
    dynamic_state[..., 15] = velocity[..., 0]       # ut
    dynamic_state[..., 16] = displacement[..., 1]  # v
    dynamic_state[..., 17] = velocity[..., 1]       # vt
    dynamic_state[..., 18] = displacement[..., 2]  # w  — ~0 in plane-strain
    dynamic_state[..., 19] = velocity[..., 2]       # wt — ~0 in plane-strain
    node_static_raw, static_names = build_node_static(coords, scenario)
    thermal_static = thermal_properties[:, 0, :]
    node_material_raw = np.column_stack(
        [
            material_static[:, 0],
            material_static[:, 1],
            material_static[:, 2],
            material_static[:, 3],
            thermal_static[:, 0],
            thermal_static[:, 2],
        ]
    ).astype(np.float32)
    node_static_raw = np.concatenate([node_static_raw, node_material_raw], axis=1).astype(np.float32)
    static_names = static_names + MATERIAL_FEATURE_NAMES
    # Pad with a single zero column so node_in_dim matches the
    # sandstone_comsol_real checkpoint exactly (node_in_dim=80,
    # out_dim=20). The padded feature is constant 0 across all nodes
    # and does not influence the trained MLP because the corresponding
    # weight column was learned on a feature whose data we don't have.
    n_nodes = node_static_raw.shape[0]
    pad_col = np.zeros((n_nodes, 1), dtype=np.float32)
    node_static_raw = np.concatenate([node_static_raw, pad_col], axis=1).astype(np.float32)
    static_names = static_names + ["checkpoint_pad_0"]

    dyn_norm = FeatureNormalizer(min_std=min_std).fit(DYNAMIC_FIELD_NAMES, dynamic_state)
    static_norm = FeatureNormalizer(min_std=min_std).fit(static_names, node_static_raw)
    dynamic_state_norm = dyn_norm.normalize_array(DYNAMIC_FIELD_NAMES, dynamic_state)
    node_static_norm = static_norm.normalize_array(static_names, node_static_raw)

    edge_index, edge_attr = build_graph_from_mesh(coords, {}, k_nearest)
    samples: list[tuple[torch.Tensor, torch.Tensor]] = []
    for time_index in range(len(times) - 1):
        x = np.concatenate([node_static_norm, dynamic_state_norm[time_index]], axis=1).astype(np.float32)
        if target_mode == "delta":
            y = (dynamic_state_norm[time_index + 1] - dynamic_state_norm[time_index]).astype(np.float32)
        elif target_mode == "absolute":
            y = dynamic_state_norm[time_index + 1].astype(np.float32)
        else:
            raise ValueError("target_mode must be 'delta' or 'absolute'")
        samples.append(
            (
                torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(y)),
            )
        )

    train_slice, val_slice, test_slice = split_samples(len(times), train_ratio, val_ratio)
    data = {
        "train": samples[train_slice],
        "val": samples[val_slice],
        "test": samples[test_slice],
    }

    graph = {
        "coords": torch.from_numpy(np.ascontiguousarray(coords.astype(np.float32))),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "node_static": torch.from_numpy(np.ascontiguousarray(node_static_norm.astype(np.float32))),
        "node_static_raw": torch.from_numpy(np.ascontiguousarray(node_static_raw.astype(np.float32))),
        "static_feature_names": static_names,
    }

    metadata = {
        "dataset_id": dataset_id,
        "rock_type": rock,
        "raw_dir": "",
        "mesh_file": "",
        "graph_source": "structured_2d_knn",
        "source_structured_dataset": str(source_structured_dataset),
        "source_metadata": str(source_metadata_path),
        "n_nodes": int(coords.shape[0]),
        "n_edges": int(edge_index.shape[1]),
        "n_timesteps": int(len(times)),
        "times": [float(t) for t in times.tolist()],
        "field_names": list(DYNAMIC_FIELD_NAMES),
        "field_units": dict(DYNAMIC_FIELD_UNITS),
        "static_feature_names": static_names,
        "node_material_feature_names": list(MATERIAL_FEATURE_NAMES),
        "node_material_units": {
            "young_modulus": "Pa",
            "poisson_ratio": "1",
            "density": "kg/m^3",
            "thermal_expansion": "1/K",
            "thermal_conductivity": "W/(m*K)",
            "heat_capacity": "J/(kg*K)",
        },
        "n_dynamic_fields": len(DYNAMIC_FIELD_NAMES),
        "n_static_features": len(static_names),
        "n_mask_features": 0,
        "node_in_dim": len(static_names) + len(DYNAMIC_FIELD_NAMES),
        "edge_in_dim": int(edge_attr.shape[1]),
        "target_mode": target_mode,
        "n_train": train_slice.stop - train_slice.start,
        "n_val": val_slice.stop - val_slice.start,
        "n_test": test_slice.stop - test_slice.start,
        "scenario": scenario,
        "dimension": 2,
        "effective_domain_type": "rect_2d",
    }
    normalization = {"dynamic": dyn_norm.to_dict(), "static": static_norm.to_dict()}
    return graph, data, metadata, normalization


def split_samples(time_steps: int, train_ratio: float, val_ratio: float) -> tuple[slice, slice, slice]:
    total_pairs = max(time_steps - 1, 0)
    train_size = max(1, int(total_pairs * train_ratio)) if total_pairs > 0 else 0
    val_size = max(1, int(total_pairs * val_ratio)) if total_pairs - train_size > 1 else max(0, total_pairs - train_size)
    train_size = min(train_size, total_pairs)
    val_size = min(val_size, max(0, total_pairs - train_size))
    test_size = max(0, total_pairs - train_size - val_size)
    return (
        slice(0, train_size),
        slice(train_size, train_size + val_size),
        slice(train_size + val_size, train_size + val_size + test_size),
    )


def write_preview_csv(path: Path, field_names: list[str], field_units: dict[str, str], payload: np.lib.npyio.NpzFile) -> None:
    disp = payload["displacement"].astype(np.float32)
    vel = payload["velocity"].astype(np.float32)
    sn = payload["stress_normal"].astype(np.float32)  # (xx, yy, zz, mises)
    ss = payload["stress_shear"].astype(np.float32)   # (xy, xz, yz)
    eps = payload["strain"].astype(np.float32)        # (xx, yy, zz, xy, xz, yz)
    dynamic_arrays = {
        # 7 legacy keys kept for backward compat
        "temperature": payload["temperature"].astype(np.float32),
        "u": disp[:, :, 0], "v": disp[:, :, 1], "w": disp[:, :, 2],
        "ut": vel[:, :, 0], "vt": vel[:, :, 1], "wt": vel[:, :, 2],
        # 20-field schema (checkpoint metadata field_names)
        "solid.ex": eps[:, :, 0], "solid.ey": eps[:, :, 1], "solid.ez": eps[:, :, 2],
        "solid.exy": eps[:, :, 3], "solid.exz": eps[:, :, 4], "solid.eyz": eps[:, :, 5],
        "solid.mises": sn[:, :, 3],
        "solid.sx": sn[:, :, 0], "solid.sy": sn[:, :, 1], "solid.sz": sn[:, :, 2],
        "solid.sxy": ss[:, :, 0], "solid.sxz": ss[:, :, 1], "solid.syz": ss[:, :, 2],
        "t": payload["temperature"].astype(np.float32),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["field", "unit", "min", "max", "mean", "std"])
        writer.writeheader()
        for name in field_names:
            values = dynamic_arrays[name]
            writer.writerow(
                {
                    "field": name,
                    "unit": field_units.get(name, ""),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
            )


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


if __name__ == "__main__":
    main()
