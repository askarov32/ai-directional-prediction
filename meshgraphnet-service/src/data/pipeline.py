"""Commercial data preparation pipeline: real COMSOL -> graph trajectories."""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from .comsol_reader import align_coordinates, build_state_tensor, load_raw_csvs, extract_material_metadata, infer_temperature_source, extract_node_static_material_fields
from .dataset_registry import dataset_dir, load_scenario, resolve_raw_and_mesh
from .graph_builder import build_graph_from_mesh
from .mesh_parser import parse_mphtxt
from .normalizer import FeatureNormalizer
from .scenario import scenario_feature_vector, source_center_radius_temperature, save_yaml, normalize_scenario_schema


def build_node_static(coords: np.ndarray, scenario: Dict) -> Tuple[np.ndarray, List[str]]:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.shape[1] == 2:
        coords = np.column_stack([coords, np.zeros(len(coords), dtype=np.float32)])
    sc_vec, sc_names = scenario_feature_vector(scenario)
    center, radius, _, _ = source_center_radius_temperature(scenario)
    dist = np.linalg.norm(coords[:, :3] - center.reshape(1, 3), axis=1, keepdims=True).astype(np.float32)
    inside = (dist[:, 0] <= max(radius, 0.0)).astype(np.float32).reshape(-1, 1)
    static = np.concatenate(
        [coords[:, :3], dist, inside, np.repeat(sc_vec.reshape(1, -1), len(coords), axis=0)], axis=1
    ).astype(np.float32)
    names = ["x", "y", "z", "distance_to_source", "is_source_region"] + sc_names
    return static, names


def _split_samples(T: int, train_ratio: float, val_ratio: float) -> tuple[slice, slice, slice]:
    n = max(T - 1, 0)
    n_train = max(1, int(n * train_ratio)) if n > 0 else 0
    n_val = max(1, int(n * val_ratio)) if n - n_train > 1 else max(0, n - n_train)
    n_train = min(n_train, n)
    n_val = min(n_val, max(0, n - n_train))
    n_test = max(0, n - n_train - n_val)
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n_train + n_val + n_test)


def run_pipeline(dataset_id: str, config: Dict | None = None, registry_dir: str | Path = "datasets") -> Dict:
    config = config or {}
    reg = Path(config.get("registry_dir", registry_dir))
    d = dataset_dir(dataset_id, reg)
    processed = d / config.get("processed_subdir", "processed")
    processed.mkdir(parents=True, exist_ok=True)

    scenario = normalize_scenario_schema(load_scenario(dataset_id, reg), dataset_id=dataset_id)
    raw_dir, mesh_file = resolve_raw_and_mesh(dataset_id, reg)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found for {dataset_id}: {raw_dir}")

    csv_dict = load_raw_csvs(raw_dir)
    csv_coords = align_coordinates(csv_dict)

    # Enrich scenario.yaml from real COMSOL exports when values are missing.
    # Material fields are intentionally used as static scenario metadata, not as
    # dynamic targets for the network.
    inferred_material = extract_material_metadata(csv_dict)
    material = scenario.setdefault("material", {})
    for key, value in inferred_material.items():
        if value is not None and (material.get(key, 0.0) in [0, 0.0, None, ""]):
            material[key] = float(value)

    inferred_source = infer_temperature_source(csv_dict)
    source = scenario.setdefault("source", {})
    scn = scenario.setdefault("scenario", {})
    if "initial_temperature" in inferred_source and source.get("initial_temperature", 0.0) in [0, 0.0, None, ""]:
        source["initial_temperature"] = float(inferred_source["initial_temperature"])
        if float(scn.get("initial_temperature", 0.0) or 0.0) == 0.0:
            scn["initial_temperature"] = float(inferred_source["initial_temperature"])
    if "background_temperature" in inferred_source and source.get("background_temperature", 0.0) in [0, 0.0, None, ""]:
        source["background_temperature"] = float(inferred_source["background_temperature"])
        if float(scn.get("background_temperature", 0.0) or 0.0) == 0.0:
            scn["background_temperature"] = float(inferred_source["background_temperature"])
    if "center" in inferred_source and source.get("center", [0, 0, 0]) in ([0, 0, 0], [0.0, 0.0, 0.0], None):
        source["center"] = inferred_source["center"]
        if scn.get("source_center") in ([0, 0, 0], [0.0, 0.0, 0.0], None):
            scn["source_center"] = inferred_source["center"]
    if "radius" in inferred_source and source.get("radius", 0.0) in [0, 0.0, None, ""]:
        source["radius"] = float(inferred_source["radius"])
        if float(scn.get("source_radius", 0.0) or 0.0) == 0.0:
            scn["source_radius"] = float(inferred_source["radius"])
    scenario = normalize_scenario_schema(scenario, dataset_id=dataset_id)

    state_raw, field_names, times, units = build_state_tensor(csv_dict, reference_coords=csv_coords)  # [T,N,F]

    # Fill time metadata when absent.
    time_meta = scenario.setdefault("time", {})
    if len(times) >= 2 and time_meta.get("step", 0.0) in [0, 0.0, None, ""]:
        time_meta["step"] = float(times[1] - times[0])
    if times and time_meta.get("start", 0.0) in [0, 0.0, None, ""]:
        time_meta["start"] = float(times[0])
    if times and time_meta.get("end", 0.0) in [0, 0.0, None, ""]:
        time_meta["end"] = float(times[-1])

    coords = csv_coords
    elements = {}
    graph_source = "knn_fallback"
    if mesh_file is not None and Path(mesh_file).exists() and Path(mesh_file).is_file() and bool(config.get("use_mesh_edges", True)):
        try:
            mesh_coords, elements = parse_mphtxt(mesh_file)
            graph_source = "mesh"
            # Use CSV coordinates as data nodes. This is usually correct for exported point data.
            if len(mesh_coords) != len(csv_coords):
                warnings.warn(
                    f"Mesh nodes ({len(mesh_coords)}) != CSV rows ({len(csv_coords)}). Using CSV coordinates and kNN fallback graph nodes."
                )
                elements = {}
                graph_source = "knn_fallback_mesh_csv_node_mismatch"
        except Exception as exc:
            warnings.warn(f"Could not parse mesh file {mesh_file}: {exc}. Falling back to kNN graph from CSV coordinates.")
            elements = {}
            graph_source = "knn_fallback_mesh_parse_error"
    else:
        warnings.warn("No valid .mphtxt/.mphbin mesh file found or mesh edges disabled. Building kNN graph from CSV coordinates.")

    edge_index, edge_attr = build_graph_from_mesh(coords, elements, int(config.get("k_nearest", 12)))
    node_static_raw, static_names = build_node_static(coords, scenario)
    node_material_raw, node_material_names, node_material_units = extract_node_static_material_fields(csv_dict, coords)
    if node_material_raw.shape[1] > 0:
        node_static_raw = np.concatenate([node_static_raw, node_material_raw], axis=1).astype(np.float32)
        static_names = static_names + node_material_names

    dyn_norm = FeatureNormalizer(min_std=float(config.get("min_std", 1e-8))).fit(field_names, state_raw)
    static_norm = FeatureNormalizer(min_std=float(config.get("min_std", 1e-8))).fit(static_names, node_static_raw)
    state_norm = dyn_norm.normalize_array(field_names, state_raw)
    node_static = static_norm.normalize_array(static_names, node_static_raw)

    target_mode = scenario.get("training", {}).get("target_mode", config.get("target_mode", "delta"))
    samples = []
    for t in range(len(times) - 1):
        x = np.concatenate([node_static, state_norm[t]], axis=1).astype(np.float32)
        if target_mode == "delta":
            y = (state_norm[t + 1] - state_norm[t]).astype(np.float32)
        elif target_mode == "absolute":
            y = state_norm[t + 1].astype(np.float32)
        else:
            raise ValueError("target_mode must be 'delta' or 'absolute'")
        samples.append((torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))))

    train_ratio = float(scenario.get("training", {}).get("train_ratio", config.get("train_ratio", 0.7)))
    val_ratio = float(scenario.get("training", {}).get("val_ratio", config.get("val_ratio", 0.15)))
    tr, va, te = _split_samples(len(times), train_ratio, val_ratio)
    data = {"train": samples[tr], "val": samples[va], "test": samples[te]}

    graph = {
        "coords": torch.from_numpy(np.ascontiguousarray(coords.astype(np.float32))),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "node_static": torch.from_numpy(np.ascontiguousarray(node_static.astype(np.float32))),
        "node_static_raw": torch.from_numpy(np.ascontiguousarray(node_static_raw.astype(np.float32))),
        "static_feature_names": static_names,
    }
    # Persist the enriched scenario so future UI/training runs show real material/time values.
    save_yaml(scenario, d / "scenario.yaml")

    metadata = {
        "dataset_id": dataset_id,
        "raw_dir": str(raw_dir),
        "mesh_file": str(mesh_file) if mesh_file is not None else "",
        "graph_source": graph_source,
        "n_nodes": int(len(coords)),
        "n_edges": int(edge_index.shape[1]),
        "n_timesteps": int(len(times)),
        "times": [float(t) for t in times],
        "field_names": field_names,
        "field_units": units,
        "static_feature_names": static_names,
        "node_material_feature_names": node_material_names,
        "node_material_units": node_material_units,
        "n_dynamic_fields": len(field_names),
        "node_in_dim": len(static_names) + len(field_names),
        "edge_in_dim": int(edge_attr.shape[1]),
        "target_mode": target_mode,
        "n_train": len(data["train"]),
        "n_val": len(data["val"]),
        "n_test": len(data["test"]),
        "scenario": scenario,
    }

    torch.save(graph, processed / "graph.pt")
    torch.save(data, processed / "trajectories.pt")
    dyn_norm.save(processed / "dynamic_normalization.json")
    static_norm.save(processed / "static_normalization.json")
    # Combined file for convenience.
    with (processed / "normalization.json").open("w", encoding="utf-8") as f:
        json.dump({"dynamic": dyn_norm.to_dict(), "static": static_norm.to_dict()}, f, indent=2, ensure_ascii=False)
    with (processed / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    preview_rows = []
    for i, f_name in enumerate(field_names):
        v = state_raw[:, :, i]
        preview_rows.append(
            {
                "field": f_name,
                "unit": units.get(f_name, ""),
                "min": float(np.min(v)),
                "max": float(np.max(v)),
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
            }
        )
    pd.DataFrame(preview_rows).to_csv(processed / "preview.csv", index=False)
    return metadata


def load_processed_dataset(dataset_id: str, registry_dir: str | Path = "datasets") -> Dict:
    d = dataset_dir(dataset_id, registry_dir) / "processed"
    if not (d / "metadata.json").exists():
        raise FileNotFoundError(f"Dataset is not processed: {dataset_id}. Run prepare_dataset.py first.")
    graph = torch.load(d / "graph.pt", map_location="cpu")
    data = torch.load(d / "trajectories.pt", map_location="cpu")
    with (d / "metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    with (d / "normalization.json").open("r", encoding="utf-8") as f:
        norm = json.load(f)
    return {"graph": graph, "data": data, "metadata": metadata, "normalization": norm, "processed_dir": d}
