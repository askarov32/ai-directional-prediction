"""Autoregressive rollout for Conditional MeshGraphNet."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from src.data.normalizer import FeatureNormalizer
from src.data.pipeline import load_processed_dataset
from src.data.scenario import source_center_radius_temperature
from src.models.meshgraphnet import ConditionalMeshGraphNet
from src.training.checkpoint_manager import load_checkpoint
from src.training.train import build_model, setup_device


def _temperature_index(field_names):
    for i, f in enumerate(field_names):
        fl = f.lower()
        if fl in {"t", "temp", "temperature"} or "temperature" in fl:
            return i
    return None


def build_initial_state(ds: Dict, config: Dict) -> torch.Tensor:
    metadata = ds["metadata"]
    graph = ds["graph"]
    field_names = metadata["field_names"]
    F = len(field_names)
    S = metadata["node_in_dim"] - F
    inf = config.get("inference", {})
    source = inf.get("initial_state_source", "from_dataset")

    if source in {"from_dataset", "nearest_scenario"}:
        # nearest_scenario currently falls back to first available real state.
        for split in ["test", "val", "train"]:
            samples = ds["data"].get(split, [])
            if samples:
                return samples[0][0].clone()
        raise ValueError("No samples available to build initial state.")

    if source == "user_defined":
        static = graph["node_static"].clone()
        dynamic_raw = np.zeros((metadata["n_nodes"], F), dtype=np.float32)
        scenario = metadata.get("scenario", {})
        scenario.update({"source": scenario.get("source", {})})
        # override from inference config
        sc_inf = inf.get("scenario", {}) or {}
        if sc_inf:
            scenario.setdefault("source", {})
            scenario["source"]["initial_temperature"] = sc_inf.get("initial_temperature", scenario["source"].get("initial_temperature", 0.0))
            scenario["source"]["background_temperature"] = sc_inf.get("background_temperature", scenario["source"].get("background_temperature", 0.0))
            scenario["source"]["radius"] = sc_inf.get("source_radius", scenario["source"].get("radius", 0.0))
            scenario["source"]["center"] = sc_inf.get("source_center", scenario["source"].get("center", [0, 0, 0]))
        center, radius, t_src, t_bg = source_center_radius_temperature(scenario)
        t_idx = _temperature_index(field_names)
        if t_idx is not None:
            coords = graph["coords"].numpy()
            dist = np.linalg.norm(coords[:, :3] - center.reshape(1, 3), axis=1)
            dynamic_raw[:, t_idx] = t_bg
            dynamic_raw[dist <= radius, t_idx] = t_src
        dyn_norm = FeatureNormalizer.from_dict(ds["normalization"].get("dynamic", {}))
        dynamic = torch.tensor(dyn_norm.normalize_array(field_names, dynamic_raw), dtype=torch.float32)
        return torch.cat([static, dynamic], dim=1)

    raise ValueError("initial_state_source must be from_dataset | nearest_scenario | user_defined")


def run_rollout(config: Dict, dataset_id: str | None = None, checkpoint: str | None = None, rollout_steps: int | None = None) -> Tuple[np.ndarray, np.ndarray, list, list, Dict]:
    inf = config.get("inference", {})
    data_cfg = config.get("data", {})
    dataset_id = dataset_id or inf.get("dataset_id")
    checkpoint = checkpoint or inf.get("checkpoint")
    steps = int(rollout_steps or inf.get("rollout_steps", 100))
    if not dataset_id:
        raise ValueError("dataset_id is required")
    if not checkpoint:
        raise ValueError("checkpoint is required")

    device = setup_device(inf.get("device", config.get("training", {}).get("device", "auto")))
    ds = load_processed_dataset(dataset_id, data_cfg.get("registry_dir", "datasets"))
    md = ds["metadata"]
    model = build_model(config, md["node_in_dim"], md["edge_in_dim"], len(md["field_names"])).to(device)
    load_checkpoint(checkpoint, model, map_location=device, strict=True)
    model.eval()

    edge_index = ds["graph"]["edge_index"].to(device)
    edge_attr = ds["graph"]["edge_attr"].to(device)
    x = build_initial_state(ds, config).to(device)
    F = len(md["field_names"])
    S = md["node_in_dim"] - F
    target_mode = md.get("target_mode", config.get("data", {}).get("target_mode", "delta"))
    states_norm = [x[:, S:].detach().cpu().numpy()]

    with torch.no_grad():
        for _ in range(steps):
            pred = model(x, edge_index, edge_attr)
            current = x[:, S:]
            if target_mode == "delta":
                next_state = current + pred
            else:
                next_state = pred
            x = torch.cat([x[:, :S], next_state], dim=1)
            states_norm.append(next_state.detach().cpu().numpy())

    dyn_norm = FeatureNormalizer.from_dict(ds["normalization"].get("dynamic", {}))
    traj_raw = dyn_norm.denormalize_array(md["field_names"], np.asarray(states_norm, dtype=np.float32))
    coords = ds["graph"]["coords"].detach().cpu().numpy()
    # Build synthetic predicted times from known dt/first time spacing.
    times = md.get("times", [])
    if len(times) >= 2:
        dt = float(times[1] - times[0])
        start = float(times[0])
    else:
        dt = float(md.get("scenario", {}).get("time", {}).get("step", 1.0) or 1.0)
        start = 0.0
    pred_times = [start + i * dt for i in range(len(traj_raw))]
    return traj_raw, coords, md["field_names"], pred_times, md
