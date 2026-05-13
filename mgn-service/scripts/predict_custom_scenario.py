from __future__ import annotations

"""Prediction on the same geometry with overridden rock and temperature parameters.

This script keeps the processed graph topology fixed:
    coords, edge_index, edge_attr
and changes only:
    1) node_static material/scenario features
    2) initial temperature field in state_0

Typical use:
python scripts/predict_custom_scenario.py ^
  --config configs/inference.yaml ^
  --dataset_id limestone_comsol_real ^
  --checkpoint outputs/checkpoints/best_model.pt ^
  --initial_temperature 873.15 ^
  --background_temperature 293.15 ^
  --source_center 0.5 0.5 0.5 ^
  --source_radius 0.05 ^
  --young_modulus 5e10 ^
  --poisson_ratio 0.25 ^
  --density 2600 ^
  --thermal_expansion 8e-6 ^
  --thermal_conductivity 2.5 ^
  --heat_capacity 900 ^
  --rollout_steps 100
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.data.normalizer import FeatureNormalizer
from src.data.pipeline import load_processed_dataset
from src.inference.export import compute_derived_fields, export_prediction
from src.models.meshgraphnet import ConditionalMeshGraphNet  # noqa: F401 - useful for torch checkpoints
from src.training.checkpoint_manager import load_checkpoint
from src.training.train import build_model, load_config, setup_device


MAT_ARG_TO_STATIC_NAMES = {
    "young_modulus": ["mat_young_modulus", "node_solid_e"],
    "poisson_ratio": ["mat_poisson_ratio", "node_solid_nu"],
    "density": ["mat_density", "node_solid_rho", "node_ht_rho"],
    "thermal_expansion": ["mat_thermal_expansion", "node_solid_alpha_iso", "node_te1_alpha_iso"],
    "thermal_conductivity": ["mat_thermal_conductivity", "node_ht_k_iso"],
    "heat_capacity": ["mat_heat_capacity", "node_ht_cp"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run MeshGraphNet prediction on the same geometry with custom rock/temperature parameters."
    )
    p.add_argument("--config", default="configs/inference.yaml")
    p.add_argument("--dataset_id", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--rollout_steps", type=int, default=None)
    p.add_argument("--output_dir", default="outputs/predictions_custom")
    p.add_argument("--device", default=None)

    # Temperature/source override.
    p.add_argument("--initial_temperature", type=float, default=None, help="Temperature inside source region, K.")
    p.add_argument("--background_temperature", type=float, default=None, help="Temperature outside source region, K.")
    p.add_argument("--source_center", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"))
    p.add_argument("--source_radius", type=float, default=None)
    p.add_argument(
        "--temperature_mode",
        choices=["source_region", "uniform_background", "keep_dataset"],
        default="source_region",
        help=(
            "source_region: set background everywhere and initial_temperature inside source radius; "
            "uniform_background: set all T to background_temperature; "
            "keep_dataset: do not change T."
        ),
    )

    # Rock/material override. Units must match COMSOL export/training data.
    p.add_argument("--young_modulus", type=float, default=None, help="E / Young modulus.")
    p.add_argument("--poisson_ratio", type=float, default=None, help="nu.")
    p.add_argument("--density", type=float, default=None, help="rho.")
    p.add_argument("--thermal_expansion", type=float, default=None, help="alpha.")
    p.add_argument("--thermal_conductivity", type=float, default=None, help="k.")
    p.add_argument("--heat_capacity", type=float, default=None, help="Cp.")
    p.add_argument(
        "--set_static",
        action="append",
        default=[],
        help="Manual static override as feature=value. Can be repeated, e.g. --set_static node_solid_e=5e10.",
    )
    p.add_argument("--save_custom_graph", action="store_true", help="Save graph_custom.pt with overridden node_static.")
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--no_animate", action="store_true")
    p.add_argument("--no_vtk", action="store_true")
    return p.parse_args()


def _temperature_index(field_names: List[str]) -> Optional[int]:
    for i, f in enumerate(field_names):
        fl = str(f).lower()
        if fl in {"t", "temp", "temperature"} or "temperature" in fl:
            return i
    return None


def _load_initial_dynamic_norm(ds: Dict, static_dim: int) -> torch.Tensor:
    """Return dynamic part of the first available sample, normalized, shape [N,F]."""
    for split in ["test", "val", "train"]:
        samples = ds["data"].get(split, [])
        if samples:
            x0 = samples[0][0]
            return x0[:, static_dim:].clone().float()
    raise ValueError("No samples found in trajectories.pt. Cannot build initial state.")


def _to_raw_static(graph: Dict, static_names: List[str], normalization: Dict) -> np.ndarray:
    if "node_static_raw" in graph:
        return graph["node_static_raw"].detach().cpu().numpy().astype(np.float32).copy()
    static_norm = graph["node_static"].detach().cpu().numpy().astype(np.float32)
    static_normalizer = FeatureNormalizer.from_dict(normalization.get("static", {}))
    return static_normalizer.denormalize_array(static_names, static_norm)


def _set_feature(raw: np.ndarray, names: List[str], name: str, value: float, *, verbose: bool = True) -> bool:
    if name not in names:
        if verbose:
            print(f"⚠️ static feature not found, skipped: {name}")
        return False
    idx = names.index(name)
    raw[:, idx] = float(value)
    return True


def _parse_manual_static(pairs: Iterable[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"Invalid --set_static value: {item}. Expected feature=value")
        key, val = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --set_static key in: {item}")
        out[key] = float(val)
    return out


def apply_static_overrides(
    graph: Dict,
    metadata: Dict,
    normalization: Dict,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Create normalized node_static after material/source overrides."""
    coords = graph["coords"].detach().cpu().numpy().astype(np.float32)
    static_names = list(metadata["static_feature_names"])
    raw = _to_raw_static(graph, static_names, normalization)
    applied: Dict[str, object] = {}

    # Source geometry affects distance_to_source and is_source_region.
    center = np.asarray(args.source_center if args.source_center is not None else None, dtype=np.float32) if args.source_center is not None else None
    radius = float(args.source_radius) if args.source_radius is not None else None
    if center is not None:
        for name, value in [
            ("heated_source_center_x", center[0]),
            ("heated_source_center_y", center[1]),
            ("heated_source_center_z", center[2]),
        ]:
            _set_feature(raw, static_names, name, float(value), verbose=False)
        applied["source_center"] = [float(x) for x in center]
    if radius is not None:
        _set_feature(raw, static_names, "heated_source_radius", radius, verbose=False)
        applied["source_radius"] = radius
    if center is not None or radius is not None:
        # If one parameter is absent, recover it from current raw static when possible.
        if center is None:
            if all(n in static_names for n in ["heated_source_center_x", "heated_source_center_y", "heated_source_center_z"]):
                center = np.asarray([
                    raw[0, static_names.index("heated_source_center_x")],
                    raw[0, static_names.index("heated_source_center_y")],
                    raw[0, static_names.index("heated_source_center_z")],
                ], dtype=np.float32)
            else:
                center = coords.mean(axis=0).astype(np.float32)
        if radius is None:
            if "heated_source_radius" in static_names:
                radius = float(raw[0, static_names.index("heated_source_radius")])
            else:
                radius = 0.0
        dist = np.linalg.norm(coords[:, :3] - center.reshape(1, 3), axis=1).astype(np.float32)
        if "distance_to_source" in static_names:
            raw[:, static_names.index("distance_to_source")] = dist
        if "is_source_region" in static_names:
            raw[:, static_names.index("is_source_region")] = (dist <= max(float(radius), 0.0)).astype(np.float32)

    # Scenario temperature scalar features.
    if args.initial_temperature is not None:
        _set_feature(raw, static_names, "heated_initial_temperature", args.initial_temperature, verbose=False)
        applied["initial_temperature"] = float(args.initial_temperature)
    if args.background_temperature is not None:
        _set_feature(raw, static_names, "heated_background_temperature", args.background_temperature, verbose=False)
        applied["background_temperature"] = float(args.background_temperature)

    # Material features. We update both global scenario slots and nodewise COMSOL material slots if present.
    for arg_name, feature_names in MAT_ARG_TO_STATIC_NAMES.items():
        value = getattr(args, arg_name, None)
        if value is None:
            continue
        applied[arg_name] = float(value)
        for feature_name in feature_names:
            _set_feature(raw, static_names, feature_name, value, verbose=False)

    # Manual exact-feature overrides for project-specific names.
    manual = _parse_manual_static(args.set_static)
    for name, value in manual.items():
        ok = _set_feature(raw, static_names, name, value, verbose=True)
        if ok:
            applied[f"static:{name}"] = float(value)

    static_normalizer = FeatureNormalizer.from_dict(normalization.get("static", {}))
    norm = static_normalizer.normalize_array(static_names, raw).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(norm)), applied


def apply_temperature_override(
    dynamic_norm: torch.Tensor,
    graph: Dict,
    metadata: Dict,
    normalization: Dict,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Return normalized dynamic state after initial temperature override."""
    field_names = list(metadata["field_names"])
    t_idx = _temperature_index(field_names)
    if args.temperature_mode == "keep_dataset":
        return dynamic_norm.clone(), {"temperature_mode": "keep_dataset"}
    if t_idx is None:
        print("⚠️ No temperature field found in field_names; temperature override skipped.")
        return dynamic_norm.clone(), {"temperature_mode": "skipped_no_temperature_field"}

    dyn_normalizer = FeatureNormalizer.from_dict(normalization.get("dynamic", {}))
    dynamic_raw = dyn_normalizer.denormalize_array(field_names, dynamic_norm.detach().cpu().numpy().astype(np.float32))

    coords = graph["coords"].detach().cpu().numpy().astype(np.float32)
    if args.temperature_mode == "uniform_background":
        if args.background_temperature is None:
            raise ValueError("--temperature_mode uniform_background requires --background_temperature")
        dynamic_raw[:, t_idx] = float(args.background_temperature)
        info = {"temperature_mode": "uniform_background", "background_temperature": float(args.background_temperature)}
    else:
        if args.initial_temperature is None or args.background_temperature is None:
            raise ValueError("--temperature_mode source_region requires --initial_temperature and --background_temperature")
        if args.source_center is None or args.source_radius is None:
            raise ValueError("--temperature_mode source_region requires --source_center X Y Z and --source_radius")
        center = np.asarray(args.source_center, dtype=np.float32)
        radius = float(args.source_radius)
        dist = np.linalg.norm(coords[:, :3] - center.reshape(1, 3), axis=1)
        mask = dist <= max(radius, 0.0)
        dynamic_raw[:, t_idx] = float(args.background_temperature)
        dynamic_raw[mask, t_idx] = float(args.initial_temperature)
        info = {
            "temperature_mode": "source_region",
            "temperature_field": field_names[t_idx],
            "initial_temperature": float(args.initial_temperature),
            "background_temperature": float(args.background_temperature),
            "source_center": [float(x) for x in center],
            "source_radius": radius,
            "source_nodes": int(mask.sum()),
        }
    dynamic_norm_new = dyn_normalizer.normalize_array(field_names, dynamic_raw).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(dynamic_norm_new)), info


def main() -> None:
    args = parse_args()
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    cfg = load_config(args.config)
    inf = cfg.get("inference", {})
    steps = int(args.rollout_steps or inf.get("rollout_steps", 100))
    device = setup_device(args.device or inf.get("device", cfg.get("training", {}).get("device", "auto")))

    ds = load_processed_dataset(args.dataset_id, cfg.get("data", {}).get("registry_dir", "datasets"))
    graph = ds["graph"]
    md = ds["metadata"]
    norm = ds["normalization"]
    field_names = list(md["field_names"])
    static_names = list(md["static_feature_names"])
    F = len(field_names)
    S = len(static_names)

    if md.get("node_in_dim") != S + F:
        raise ValueError(f"metadata node_in_dim mismatch: {md.get('node_in_dim')} != {S}+{F}")

    # Build model exactly as during training.
    model = build_model(cfg, md["node_in_dim"], md["edge_in_dim"], F).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device, strict=True)
    model.eval()

    node_static_norm, static_applied = apply_static_overrides(graph, md, norm, args)
    dynamic0_norm = _load_initial_dynamic_norm(ds, S)
    dynamic0_norm, temp_applied = apply_temperature_override(dynamic0_norm, graph, md, norm, args)

    x = torch.cat([node_static_norm, dynamic0_norm], dim=1).to(device)
    edge_index = graph["edge_index"].to(device)
    edge_attr = graph["edge_attr"].to(device)
    target_mode = md.get("target_mode", cfg.get("data", {}).get("target_mode", "delta"))

    states_norm = [x[:, S:].detach().cpu().numpy()]
    with torch.no_grad():
        for _ in range(steps):
            pred = model(x, edge_index, edge_attr)
            current = x[:, S:]
            next_state = current + pred if target_mode == "delta" else pred
            x = torch.cat([x[:, :S], next_state], dim=1)
            states_norm.append(next_state.detach().cpu().numpy())

    dyn_norm = FeatureNormalizer.from_dict(norm.get("dynamic", {}))
    traj_raw = dyn_norm.denormalize_array(field_names, np.asarray(states_norm, dtype=np.float32))
    coords = graph["coords"].detach().cpu().numpy()

    times = md.get("times", [])
    if len(times) >= 2:
        dt = float(times[1] - times[0])
        start = float(times[0])
    else:
        dt = float(md.get("scenario", {}).get("time", {}).get("step", 1.0) or 1.0)
        start = 0.0
    pred_times = [start + i * dt for i in range(len(traj_raw))]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    derived = compute_derived_fields(traj_raw, field_names, float(inf.get("risk_threshold", 0.8)))
    paths = export_prediction(traj_raw, coords, field_names, pred_times, derived, output_dir)

    run_meta = {
        "dataset_id": args.dataset_id,
        "checkpoint": args.checkpoint,
        "rollout_steps": steps,
        "target_mode": target_mode,
        "node_in_dim": int(md["node_in_dim"]),
        "n_nodes": int(md["n_nodes"]),
        "field_names": field_names,
        "static_overrides_applied": static_applied,
        "temperature_override_applied": temp_applied,
        "warning": (
            "This is a neural surrogate inference override, not a COMSOL solve. "
            "Trust it mainly inside the parameter range seen during training."
        ),
    }
    with (output_dir / "custom_scenario_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)

    if args.save_custom_graph:
        custom_graph = dict(graph)
        custom_graph["node_static"] = node_static_norm.cpu()
        custom_graph["custom_overrides"] = run_meta
        torch.save(custom_graph, output_dir / "graph_custom.pt")

    if not args.no_plots:
        from src.visualization.plot_3d import plot_snapshots, plot_time_series
        plot_snapshots(traj_raw, coords, field_names, derived, output_dir / "figures")
        plot_time_series(traj_raw, field_names, pred_times, output_dir / "figures")

    if inf.get("animate", True) and not args.no_animate:
        from src.visualization.animation import animate_main_fields
        animate_main_fields(traj_raw, coords, field_names, derived, output_dir / "animations")

    if inf.get("export_vtk", True) and not args.no_vtk:
        from src.visualization.vtk_export import export_vtk_sequence
        export_vtk_sequence(coords, traj_raw, field_names, derived, output_dir / "vtk")

    print("✅ Custom prediction complete")
    print(f"dataset_id: {args.dataset_id}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"n_nodes: {md['n_nodes']}")
    print(f"n_fields: {len(field_names)}")
    print(f"rollout_steps: {steps}")
    print(f"output_dir: {output_dir}")
    print("Applied overrides:")
    print(json.dumps({"static": static_applied, "temperature": temp_applied}, indent=2, ensure_ascii=False))
    print("Created:")
    for k, v in paths.items():
        print(f"{k}: {v}")
    print(f"metadata: {output_dir / 'custom_scenario_metadata.json'}")


if __name__ == "__main__":
    main()
