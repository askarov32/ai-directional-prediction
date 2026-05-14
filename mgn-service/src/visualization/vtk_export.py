"""Simple legacy VTK point-cloud export for ParaView."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np


def export_vtk_sequence(coords: np.ndarray, trajectory: np.ndarray, field_names: List[str], derived: Dict[str, np.ndarray], output_dir: str | Path, prefix: str = "prediction") -> List[str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    N = coords.shape[0]
    for t in range(trajectory.shape[0]):
        p = out / f"{prefix}_{t:04d}.vtk"
        with p.open("w", encoding="utf-8") as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write(f"MeshGraphNet prediction step {t}\n")
            f.write("ASCII\n")
            f.write("DATASET POLYDATA\n")
            f.write(f"POINTS {N} float\n")
            for c in coords:
                z = c[2] if coords.shape[1] > 2 else 0.0
                f.write(f"{float(c[0])} {float(c[1])} {float(z)}\n")
            f.write(f"VERTICES {N} {2*N}\n")
            for i in range(N):
                f.write(f"1 {i}\n")
            f.write(f"POINT_DATA {N}\n")
            for fi, name in enumerate(field_names):
                safe = name.replace(" ", "_").replace("/", "_")
                f.write(f"SCALARS {safe} float 1\nLOOKUP_TABLE default\n")
                for v in trajectory[t, :, fi]:
                    f.write(f"{float(v)}\n")
            for name, arr in derived.items():
                if arr.ndim != 2:
                    continue
                safe = name.replace(" ", "_").replace("/", "_")
                f.write(f"SCALARS {safe} float 1\nLOOKUP_TABLE default\n")
                for v in arr[t]:
                    f.write(f"{float(v)}\n")
        paths.append(str(p))
    return paths
