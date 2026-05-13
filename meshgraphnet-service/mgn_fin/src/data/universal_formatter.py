"""Universal COMSOL dataset formatter.

This module converts real COMSOL exports into a model-agnostic canonical
representation plus model-specific adapters:

raw COMSOL CSV + optional MPHTXT
    -> processed/canonical/   # source of truth: point-cloud tensors
    -> processed/graph/       # MeshGraphNet tensors
    -> processed/fno/         # optional regular grid for FNO
    -> processed/pinn/        # sampling/index metadata for PINN
    -> processed/transformer/ # token/index metadata for neural operators

The canonical representation intentionally separates dynamic target fields from
static conditioning fields. Material parameters are static features, not
prediction targets.
"""
from __future__ import annotations

import json
import math
import shutil
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

from .comsol_reader import (
    align_coordinates,
    build_state_tensor,
    extract_material_metadata,
    extract_node_static_material_fields,
    infer_temperature_source,
    load_raw_csvs,
)
from .dataset_registry import dataset_dir, load_scenario, resolve_raw_and_mesh
from .graph_builder import build_graph_from_mesh
from .mesh_parser import parse_mphtxt
from .normalizer import FeatureNormalizer
from .pipeline import build_node_static
from .scenario import normalize_scenario_schema, save_yaml, source_center_radius_temperature


def _write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _write_list(path: Path, values: Sequence[str]) -> None:
    _write_json(path, list(values))


def _split_samples(T: int, train_ratio: float, val_ratio: float) -> tuple[slice, slice, slice]:
    n = max(T - 1, 0)
    n_train = max(1, int(n * train_ratio)) if n > 0 else 0
    n_val = max(1, int(n * val_ratio)) if n - n_train > 1 else max(0, n - n_train)
    n_train = min(n_train, n)
    n_val = min(n_val, max(0, n - n_train))
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n)


def _enrich_scenario_from_data(scenario: Dict, csv_dict: Dict, times: Sequence[float] | None = None) -> Dict:
    """Fill missing material/source/time metadata from real COMSOL exports."""
    scenario = normalize_scenario_schema(scenario, dataset_id=scenario.get("dataset_id", "dataset"))

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

    time_meta = scenario.setdefault("time", {})
    if times:
        if len(times) >= 2 and time_meta.get("step", 0.0) in [0, 0.0, None, ""]:
            time_meta["step"] = float(times[1] - times[0])
        if time_meta.get("start", 0.0) in [0, 0.0, None, ""]:
            time_meta["start"] = float(times[0])
        if time_meta.get("end", 0.0) in [0, 0.0, None, ""]:
            time_meta["end"] = float(times[-1])

    return normalize_scenario_schema(scenario, dataset_id=scenario.get("dataset_id", "dataset"))


def build_masks(coords: np.ndarray, scenario: Dict, eps_fraction: float = 1e-5) -> tuple[np.ndarray, List[str]]:
    """Build explicit masks usable by every model family."""
    coords = np.asarray(coords, dtype=np.float32)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = np.maximum(maxs - mins, 1e-12)
    eps = span * eps_fraction

    masks: List[np.ndarray] = []
    names: List[str] = []

    # Domain boundary masks.
    for axis, ax in enumerate(["x", "y", "z"]):
        for side, value in [("min", mins[axis]), ("max", maxs[axis])]:
            mask = np.abs(coords[:, axis] - value) <= eps[axis]
            masks.append(mask.astype(np.float32).reshape(-1, 1))
            names.append(f"boundary_{ax}_{side}")

    sc = normalize_scenario_schema(scenario)
    stype = str(sc.get("scenario", {}).get("type", "heated_rod")).lower()
    s = sc.get("scenario", {}) or {}

    center, radius, _, _ = source_center_radius_temperature(sc)
    radius = max(float(radius), 0.0)
    dist = np.linalg.norm(coords - center.reshape(1, 3), axis=1)
    source_mask = dist <= radius if radius > 0 else np.zeros(len(coords), dtype=bool)
    masks.append(source_mask.astype(np.float32).reshape(-1, 1))
    names.append("source_mask")

    # Scenario-specific masks.
    if stype == "impact":
        center = np.asarray(s.get("impact_location", center), dtype=np.float32).reshape(3)
        radius = max(float(s.get("impact_radius", radius) or 0.0), 0.0)
        mask = np.linalg.norm(coords - center.reshape(1, 3), axis=1) <= radius if radius > 0 else np.zeros(len(coords), dtype=bool)
        masks.append(mask.astype(np.float32).reshape(-1, 1))
        names.append("impact_mask")
    else:
        masks.append(np.zeros((len(coords), 1), dtype=np.float32))
        names.append("impact_mask")

    if stype == "side_pressure":
        side = str(s.get("pressure_side", "x_min")).lower()
        axis = {"x": 0, "y": 1, "z": 2}.get(side[:1], 0)
        target = mins[axis] if side.endswith("min") else maxs[axis]
        mask = np.abs(coords[:, axis] - target) <= eps[axis]
        masks.append(mask.astype(np.float32).reshape(-1, 1))
        names.append("pressure_side_mask")
    else:
        masks.append(np.zeros((len(coords), 1), dtype=np.float32))
        names.append("pressure_side_mask")

    if stype == "building_load":
        center = np.asarray(s.get("load_area_center", center), dtype=np.float32).reshape(3)
        size = np.asarray(s.get("load_area_size", [0.0, 0.0]), dtype=np.float32).reshape(2)
        # Project load area to the top face, use x/y rectangle by default.
        top = np.abs(coords[:, 2] - maxs[2]) <= eps[2] * 10
        half = np.maximum(size / 2.0, 0.0)
        in_xy = (np.abs(coords[:, 0] - center[0]) <= half[0]) & (np.abs(coords[:, 1] - center[1]) <= half[1])
        mask = top & in_xy
        masks.append(mask.astype(np.float32).reshape(-1, 1))
        names.append("building_load_mask")
    else:
        masks.append(np.zeros((len(coords), 1), dtype=np.float32))
        names.append("building_load_mask")

    masks.append(np.ones((len(coords), 1), dtype=np.float32))
    names.append("valid_node_mask")
    return np.concatenate(masks, axis=1).astype(np.float32), names


def _save_canonical(
    canonical_dir: Path,
    coords: np.ndarray,
    times: Sequence[float],
    state_raw: np.ndarray,
    state_norm: np.ndarray,
    node_static_raw: np.ndarray,
    node_static_norm: np.ndarray,
    masks: np.ndarray,
    field_names: List[str],
    static_names: List[str],
    mask_names: List[str],
    metadata: Dict,
    normalization: Dict,
) -> None:
    canonical_dir.mkdir(parents=True, exist_ok=True)
    np.save(canonical_dir / "coords.npy", coords.astype(np.float32))
    np.save(canonical_dir / "time.npy", np.asarray(times, dtype=np.float32))
    np.save(canonical_dir / "dynamic.npy", state_raw.astype(np.float32))
    np.save(canonical_dir / "dynamic_norm.npy", state_norm.astype(np.float32))
    np.save(canonical_dir / "static.npy", node_static_raw.astype(np.float32))
    np.save(canonical_dir / "static_norm.npy", node_static_norm.astype(np.float32))
    np.save(canonical_dir / "masks.npy", masks.astype(np.float32))
    _write_list(canonical_dir / "field_names.json", field_names)
    _write_list(canonical_dir / "static_feature_names.json", static_names)
    _write_list(canonical_dir / "mask_names.json", mask_names)
    _write_json(canonical_dir / "metadata.json", metadata)
    _write_json(canonical_dir / "normalization.json", normalization)


def _save_graph(
    graph_dir: Path,
    root_processed: Path,
    coords: np.ndarray,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    node_static_raw: np.ndarray,
    node_static_norm: np.ndarray,
    static_names: List[str],
    state_norm: np.ndarray,
    field_names: List[str],
    times: Sequence[float],
    metadata: Dict,
    normalization: Dict,
    target_mode: str,
    train_ratio: float,
    val_ratio: float,
    copy_legacy: bool = True,
) -> None:
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph = {
        "coords": torch.from_numpy(np.ascontiguousarray(coords.astype(np.float32))),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "node_static": torch.from_numpy(np.ascontiguousarray(node_static_norm.astype(np.float32))),
        "node_static_raw": torch.from_numpy(np.ascontiguousarray(node_static_raw.astype(np.float32))),
        "static_feature_names": static_names,
        "field_names": field_names,
    }
    samples = []
    for t in range(len(times) - 1):
        x = np.concatenate([node_static_norm, state_norm[t]], axis=1).astype(np.float32)
        if target_mode == "delta":
            y = (state_norm[t + 1] - state_norm[t]).astype(np.float32)
        elif target_mode == "absolute":
            y = state_norm[t + 1].astype(np.float32)
        else:
            raise ValueError("target_mode must be 'delta' or 'absolute'")
        samples.append((torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))))
    tr, va, te = _split_samples(len(times), train_ratio, val_ratio)
    data = {"train": samples[tr], "val": samples[va], "test": samples[te]}

    torch.save(graph, graph_dir / "graph.pt")
    torch.save(data, graph_dir / "trajectories.pt")
    _write_json(graph_dir / "metadata.json", metadata)
    _write_json(graph_dir / "normalization.json", normalization)

    # Legacy root files preserve compatibility with existing train/predict scripts.
    if copy_legacy:
        torch.save(graph, root_processed / "graph.pt")
        torch.save(data, root_processed / "trajectories.pt")
        _write_json(root_processed / "metadata.json", metadata)
        _write_json(root_processed / "normalization.json", normalization)
        _write_json(root_processed / "dynamic_normalization.json", normalization.get("dynamic", {}))
        _write_json(root_processed / "static_normalization.json", normalization.get("static", {}))


def _regular_grid(coords: np.ndarray, resolution: Tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return grid coords as [3,Z,Y,X] and flat [M,3]. resolution = (Z,Y,X)."""
    Z, Y, X = resolution
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    xs = np.linspace(mins[0], maxs[0], X, dtype=np.float32)
    ys = np.linspace(mins[1], maxs[1], Y, dtype=np.float32)
    zs = np.linspace(mins[2], maxs[2], Z, dtype=np.float32)
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    grid_coords = np.stack([xx, yy, zz], axis=0).astype(np.float32)
    flat = np.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], axis=1).astype(np.float32)
    return grid_coords, flat


def _save_fno(
    fno_dir: Path,
    coords: np.ndarray,
    times: Sequence[float],
    state_raw: np.ndarray,
    state_norm: np.ndarray,
    node_static_raw: np.ndarray,
    node_static_norm: np.ndarray,
    masks: np.ndarray,
    field_names: List[str],
    static_names: List[str],
    mask_names: List[str],
    resolution: Tuple[int, int, int],
    max_timesteps: int | None,
    normalization_mode: str,
    metadata: Dict,
) -> None:
    """Create nearest-neighbor FNO grid.

    For large COMSOL runs, full T x C x Z x Y x X arrays can be huge.  The
    max_timesteps parameter intentionally limits exported FNO grids for first
    experiments. Canonical remains full resolution and can be regridded later.
    """
    fno_dir.mkdir(parents=True, exist_ok=True)
    grid_coords, flat = _regular_grid(coords, resolution)
    nbrs = NearestNeighbors(n_neighbors=1).fit(coords)
    dist, idx = nbrs.kneighbors(flat)
    idx = idx[:, 0].astype(np.int64)
    dist = dist[:, 0].astype(np.float32)

    span = np.maximum(coords.max(axis=0) - coords.min(axis=0), 1e-12)
    cell = float(max(span[0] / max(resolution[2] - 1, 1), span[1] / max(resolution[1] - 1, 1), span[2] / max(resolution[0] - 1, 1)))
    valid = (dist <= 2.5 * cell).astype(np.float32)

    T = state_raw.shape[0]
    if max_timesteps is not None and max_timesteps > 0 and T > max_timesteps:
        selected = np.linspace(0, T - 1, max_timesteps).round().astype(np.int64)
    else:
        selected = np.arange(T, dtype=np.int64)

    dyn_src = state_norm if normalization_mode == "normalized" else state_raw
    static_src = node_static_norm if normalization_mode == "normalized" else node_static_raw

    Z, Y, X = resolution
    grid_dynamic = dyn_src[selected][:, idx, :].reshape(len(selected), Z, Y, X, dyn_src.shape[2])
    grid_dynamic = np.transpose(grid_dynamic, (0, 4, 1, 2, 3)).astype(np.float32)
    grid_static = static_src[idx, :].reshape(Z, Y, X, static_src.shape[1])
    grid_static = np.transpose(grid_static, (3, 0, 1, 2)).astype(np.float32)
    grid_masks = masks[idx, :].reshape(Z, Y, X, masks.shape[1])
    grid_masks = np.transpose(grid_masks, (3, 0, 1, 2)).astype(np.float32)
    grid_masks = np.concatenate([grid_masks, valid.reshape(1, Z, Y, X).astype(np.float32)], axis=0)
    out_mask_names = mask_names + ["grid_valid_mask"]

    np.save(fno_dir / "grid_dynamic.npy", grid_dynamic)
    np.save(fno_dir / "grid_static.npy", grid_static)
    np.save(fno_dir / "grid_masks.npy", grid_masks)
    np.save(fno_dir / "grid_coords.npy", grid_coords)
    np.save(fno_dir / "source_node_index.npy", idx)
    np.save(fno_dir / "selected_time_indices.npy", selected)
    _write_list(fno_dir / "field_names.json", field_names)
    _write_list(fno_dir / "static_feature_names.json", static_names)
    _write_list(fno_dir / "mask_names.json", out_mask_names)
    fno_meta = dict(metadata)
    fno_meta.update(
        {
            "format": "fno_grid",
            "layout": "grid_dynamic[T,C,Z,Y,X], grid_static[S,Z,Y,X]",
            "grid_resolution_zyx": list(resolution),
            "interpolation": "nearest_node_from_comsol_mesh",
            "normalization_mode": normalization_mode,
            "selected_timesteps": int(len(selected)),
            "full_timesteps": int(T),
            "selected_time_indices_file": "selected_time_indices.npy",
        }
    )
    _write_json(fno_dir / "metadata.json", fno_meta)


def _save_index_adapter(adapter_dir: Path, metadata: Dict, adapter_name: str) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        adapter_dir / "index.json",
        {
            "format": adapter_name,
            "canonical_dir": "../canonical",
            "description": (
                "This adapter intentionally does not duplicate tensors. "
                "Use canonical/{coords,time,dynamic,static,masks}.npy and sample points/tokens on the fly."
            ),
            "n_nodes": metadata.get("n_nodes"),
            "n_timesteps": metadata.get("n_timesteps"),
            "dynamic_fields": metadata.get("field_names", []),
            "static_features": metadata.get("static_feature_names", []),
            "recommended_sampling": {
                "pinn": "sample (x,y,z,t) collocation/data points from canonical arrays",
                "transformer": "sample input/query tokens from canonical nodes and time indices",
            },
        },
    )


def run_universal_format(
    dataset_id: str,
    config: Dict | None = None,
    registry_dir: str | Path = "datasets",
    formats: Iterable[str] = ("canonical", "graph", "fno", "pinn", "transformer"),
    grid_resolution: Tuple[int, int, int] = (32, 32, 32),
    fno_max_timesteps: int | None = 128,
    fno_normalization: str = "normalized",
    copy_legacy_graph: bool = True,
) -> Dict:
    config = config or {}
    formats_set = {str(x).lower() for x in formats}
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
    state_raw, field_names, times, units = build_state_tensor(csv_dict, reference_coords=csv_coords)
    scenario = _enrich_scenario_from_data(scenario, csv_dict, times)
    save_yaml(scenario, d / "scenario.yaml")

    coords = np.asarray(csv_coords, dtype=np.float32)
    elements = {}
    graph_source = "knn_fallback"
    if mesh_file is not None and Path(mesh_file).exists() and Path(mesh_file).is_file() and bool(config.get("use_mesh_edges", True)):
        try:
            mesh_coords, elements = parse_mphtxt(mesh_file)
            graph_source = "mesh"
            if len(mesh_coords) != len(coords):
                warnings.warn(
                    f"Mesh nodes ({len(mesh_coords)}) != CSV rows ({len(coords)}). Using CSV coords + kNN fallback."
                )
                elements = {}
                graph_source = "knn_fallback_mesh_csv_node_mismatch"
        except Exception as exc:
            warnings.warn(f"Could not parse mesh file {mesh_file}: {exc}. Falling back to kNN graph.")
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

    masks, mask_names = build_masks(coords, scenario)

    dyn_norm = FeatureNormalizer(min_std=float(config.get("min_std", 1e-8))).fit(field_names, state_raw)
    static_norm = FeatureNormalizer(min_std=float(config.get("min_std", 1e-8))).fit(static_names, node_static_raw)
    state_norm = dyn_norm.normalize_array(field_names, state_raw)
    node_static_norm = static_norm.normalize_array(static_names, node_static_raw)

    target_mode = scenario.get("training", {}).get("target_mode", config.get("target_mode", "delta"))
    train_ratio = float(scenario.get("training", {}).get("train_ratio", config.get("train_ratio", 0.7)))
    val_ratio = float(scenario.get("training", {}).get("val_ratio", config.get("val_ratio", 0.15)))
    tr, va, te = _split_samples(len(times), train_ratio, val_ratio)

    normalization = {"dynamic": dyn_norm.to_dict(), "static": static_norm.to_dict()}
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
        "mask_names": mask_names,
        "node_material_feature_names": node_material_names,
        "node_material_units": node_material_units,
        "n_dynamic_fields": len(field_names),
        "n_static_features": len(static_names),
        "n_mask_features": len(mask_names),
        "node_in_dim": len(static_names) + len(field_names),
        "edge_in_dim": int(edge_attr.shape[1]),
        "target_mode": target_mode,
        "train_time_pair_range": [tr.start, tr.stop],
        "val_time_pair_range": [va.start, va.stop],
        "test_time_pair_range": [te.start, te.stop],
        "n_train": tr.stop - tr.start,
        "n_val": va.stop - va.start,
        "n_test": te.stop - te.start,
        "scenario": scenario,
        "universal_layout": {
            "canonical": "processed/canonical/{coords,time,dynamic,dynamic_norm,static,static_norm,masks}.npy",
            "meshgraphnet": "processed/graph/{graph.pt,trajectories.pt}; root legacy copies also saved",
            "fno": "processed/fno/grid_dynamic.npy [T,C,Z,Y,X] when requested",
            "pinn": "processed/pinn/index.json; sample from canonical on the fly",
            "transformer": "processed/transformer/index.json; sample tokens from canonical on the fly",
        },
    }

    if "canonical" in formats_set or "all" in formats_set:
        _save_canonical(
            processed / "canonical",
            coords,
            times,
            state_raw,
            state_norm,
            node_static_raw,
            node_static_norm,
            masks,
            field_names,
            static_names,
            mask_names,
            metadata,
            normalization,
        )

    if "graph" in formats_set or "mgn" in formats_set or "meshgraphnet" in formats_set or "all" in formats_set:
        _save_graph(
            processed / "graph",
            processed,
            coords,
            edge_index,
            edge_attr,
            node_static_raw,
            node_static_norm,
            static_names,
            state_norm,
            field_names,
            times,
            metadata,
            normalization,
            target_mode,
            train_ratio,
            val_ratio,
            copy_legacy=copy_legacy_graph,
        )

    if "fno" in formats_set or "all" in formats_set:
        if fno_normalization not in {"raw", "normalized"}:
            raise ValueError("fno_normalization must be 'raw' or 'normalized'")
        _save_fno(
            processed / "fno",
            coords,
            times,
            state_raw,
            state_norm,
            node_static_raw,
            node_static_norm,
            masks,
            field_names,
            static_names,
            mask_names,
            grid_resolution,
            fno_max_timesteps,
            fno_normalization,
            metadata,
        )

    if "pinn" in formats_set or "all" in formats_set:
        _save_index_adapter(processed / "pinn", metadata, "pinn_index")

    if "transformer" in formats_set or "neural_operator_transformer" in formats_set or "all" in formats_set:
        _save_index_adapter(processed / "transformer", metadata, "transformer_token_index")

    # Preview table is useful when checking COMSOL exports.
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
                "is_target_dynamic": True,
            }
        )
    for i, name in enumerate(static_names):
        v = node_static_raw[:, i]
        preview_rows.append(
            {
                "field": name,
                "unit": "static",
                "min": float(np.min(v)),
                "max": float(np.max(v)),
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "is_target_dynamic": False,
            }
        )
    pd.DataFrame(preview_rows).to_csv(processed / "preview.csv", index=False)
    _write_json(processed / "universal_metadata.json", metadata)
    return metadata
