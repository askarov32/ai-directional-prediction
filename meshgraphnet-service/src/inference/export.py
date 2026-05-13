"""Prediction post-processing and export."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def compute_derived_fields(trajectory: np.ndarray, field_names: List[str], risk_threshold: float = 0.8) -> Dict[str, np.ndarray]:
    fn = [f.lower() for f in field_names]
    derived: Dict[str, np.ndarray] = {}

    def idx_one(candidates):
        for cand in candidates:
            cand = cand.lower()
            for i, f in enumerate(fn):
                if f == cand or f.endswith("." + cand) or f.endswith("_" + cand):
                    return i
        return None

    disp = [idx_one([x]) for x in ["u", "v", "w"]]
    if disp[0] is not None and disp[1] is not None:
        used = [i for i in disp if i is not None]
        derived["displacement_magnitude"] = np.linalg.norm(trajectory[:, :, used], axis=2)
    vel = [idx_one([x]) for x in ["ut", "vt", "wt"]]
    if vel[0] is not None and vel[1] is not None:
        used = [i for i in vel if i is not None]
        derived["velocity_magnitude"] = np.linalg.norm(trajectory[:, :, used], axis=2)

    mises_idx = idx_one(["solid.mises", "mises", "von_mises", "von_mises_stress"])
    if mises_idx is not None:
        vm = trajectory[:, :, mises_idx]
        derived["von_mises_stress"] = vm
        vmax = np.maximum(np.nanmax(np.abs(vm), axis=1, keepdims=True), 1e-12)
        derived["risk_flag"] = (np.abs(vm) / vmax >= risk_threshold).astype(np.float32)
    else:
        sidx = [idx_one([x]) for x in ["sx", "sy", "sz", "sxy", "sxz", "syz"]]
        if all(i is not None for i in sidx):
            s = trajectory[:, :, sidx]
            sx, sy, sz, sxy, sxz, syz = [s[:, :, i] for i in range(6)]
            vm = np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2 + 6 * (sxy ** 2 + sxz ** 2 + syz ** 2)))
            derived["von_mises_stress"] = vm
            vmax = np.maximum(np.nanmax(vm, axis=1, keepdims=True), 1e-12)
            derived["risk_flag"] = (vm / vmax >= risk_threshold).astype(np.float32)

    tidx = None
    for i, f in enumerate(fn):
        if f in {"t", "temp", "temperature"} or "temperature" in f:
            tidx = i
            break
    if tidx is not None:
        temp = trajectory[:, :, tidx]
        derived["temperature_change"] = temp - temp[:1]
        derived["temperature_time_gradient"] = np.diff(temp, axis=0, prepend=temp[:1])
    return derived


def export_prediction(trajectory: np.ndarray, coords: np.ndarray, field_names: List[str], times: List[float], derived: Dict[str, np.ndarray], output_dir: str | Path = "outputs/predictions") -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save({"trajectory": torch.tensor(trajectory), "coords": torch.tensor(coords), "field_names": field_names, "times": times, "derived": {k: torch.tensor(v) for k, v in derived.items()}}, out / "prediction.pt")

    rows = []
    for ti, t in enumerate(times):
        for n in range(coords.shape[0]):
            row = {"time": t, "node": n, "x": coords[n, 0], "y": coords[n, 1], "z": coords[n, 2] if coords.shape[1] > 2 else 0.0}
            for fi, f in enumerate(field_names):
                row[f] = trajectory[ti, n, fi]
            for k, v in derived.items():
                if v.ndim == 2:
                    row[k] = v[ti, n]
            rows.append(row)
    pd.DataFrame(rows).to_csv(out / "prediction_nodes.csv", index=False)

    summary = {}
    for fi, f in enumerate(field_names):
        v = trajectory[:, :, fi]
        summary[f] = {"min": float(v.min()), "max": float(v.max()), "mean_final": float(v[-1].mean()), "max_final": float(v[-1].max())}
    for k, v in derived.items():
        summary[k] = {"min": float(v.min()), "max": float(v.max()), "mean_final": float(v[-1].mean()), "max_final": float(v[-1].max())}
    with (out / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return {"prediction_pt": str(out / "prediction.pt"), "prediction_csv": str(out / "prediction_nodes.csv"), "summary": str(out / "summary_metrics.json")}
