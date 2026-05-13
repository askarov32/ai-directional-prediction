"""GIF/MP4 animation helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import numpy as np


def animate_field(coords: np.ndarray, values_over_time: np.ndarray, field_name: str, save_path: str | Path, fps: int = 10, max_frames: int = 200) -> str:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    values = values_over_time[:max_frames]
    T = values.shape[0]
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2] if coords.shape[1] > 2 else np.zeros(len(coords))
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1.0
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, c=values[0], s=5, alpha=0.85, vmin=vmin, vmax=vmax)
    fig.colorbar(sc, ax=ax, shrink=0.65, label=field_name)
    title = ax.set_title(f"{field_name}, step 0")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    def update(frame):
        sc._offsets3d = (x, y, z)
        sc.set_array(values[frame])
        title.set_text(f"{field_name}, step {frame}")
        return (sc,)

    anim = FuncAnimation(fig, update, frames=T, interval=100, blit=False)
    try:
        if save_path.suffix.lower() == ".mp4":
            writer = FFMpegWriter(fps=fps)
        else:
            writer = PillowWriter(fps=fps)
        anim.save(str(save_path), writer=writer)
    finally:
        plt.close(fig)
    return str(save_path)


def animate_main_fields(trajectory: np.ndarray, coords: np.ndarray, field_names: List[str], derived: Dict[str, np.ndarray], output_dir: str | Path, fps: int = 10) -> List[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    preferred = []
    lower = [f.lower() for f in field_names]
    for name in ["temperature", "t", "temp"]:
        if name in lower:
            i = lower.index(name)
            preferred.append(("temperature", trajectory[:, :, i], "temperature.gif"))
            break
    for k in ["temperature_change", "displacement_magnitude", "velocity_magnitude", "von_mises_stress", "risk_flag"]:
        if k in derived:
            preferred.append((k, derived[k], f"{k}.gif"))
    for name, vals, filename in preferred:
        paths.append(animate_field(coords, vals, name, output_dir / filename, fps=fps))
    return paths
