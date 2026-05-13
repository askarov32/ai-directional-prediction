"""3D plots for fields on COMSOL nodes."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_3d_field(coords: np.ndarray, values: np.ndarray, field_name: str, title: str | None = None, save_path: str | Path | None = None) -> None:
    coords = np.asarray(coords)
    values = np.asarray(values)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    x, y = coords[:, 0], coords[:, 1]
    z = coords[:, 2] if coords.shape[1] > 2 else np.zeros(len(coords))
    sc = ax.scatter(x, y, z, c=values, s=5, alpha=0.85)
    fig.colorbar(sc, ax=ax, shrink=0.65, label=field_name)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title or field_name)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_time_series(trajectory: np.ndarray, field_names: List[str], times: List[float], output_dir: str | Path) -> List[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fi, f in enumerate(field_names):
        v = trajectory[:, :, fi]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(times, v.mean(axis=1), label="mean")
        ax.plot(times, v.max(axis=1), label="max")
        ax.plot(times, v.min(axis=1), label="min")
        ax.set_title(f"{f}: min/mean/max over time")
        ax.set_xlabel("time")
        ax.set_ylabel(f)
        ax.legend()
        p = output_dir / f"timeseries_{f}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(p))
    return paths


def plot_snapshots(trajectory: np.ndarray, coords: np.ndarray, field_names: List[str], derived: Dict[str, np.ndarray], output_dir: str | Path, timestep: int = -1) -> List[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t = timestep if timestep >= 0 else trajectory.shape[0] - 1
    paths = []
    for fi, f in enumerate(field_names):
        p = output_dir / f"snapshot_{f}_t{t}.png"
        plot_3d_field(coords, trajectory[t, :, fi], f, title=f"{f} at step {t}", save_path=p)
        paths.append(str(p))
    for k, v in derived.items():
        if v.ndim == 2:
            p = output_dir / f"snapshot_{k}_t{t}.png"
            plot_3d_field(coords, v[t], k, title=f"{k} at step {t}", save_path=p)
            paths.append(str(p))
    return paths
