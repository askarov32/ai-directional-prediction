"""COMSOL .mphtxt parser."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def parse_mphtxt(mesh_file: str | Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    mesh_file = Path(mesh_file)
    if not mesh_file.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_file}")
    lines = mesh_file.read_text(encoding="utf-8", errors="replace").splitlines()

    coords = []
    elements: Dict[str, np.ndarray] = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        low = line.lower()
        if "# number of mesh points" in low:
            try:
                n_pts = int(line.split("#")[0].strip())
            except Exception:
                i += 1
                continue
            for j in range(i + 1, min(i + 20, len(lines))):
                if "# mesh point coordinates" in lines[j].lower():
                    block = []
                    for k in range(j + 1, min(j + 1 + n_pts, len(lines))):
                        vals = lines[k].strip().split()
                        if len(vals) >= 2:
                            row = [float(v) for v in vals[:3]]
                            if len(row) == 2:
                                row.append(0.0)
                            block.append(row)
                    coords.extend(block)
                    i = j + n_pts
                    break

        # Detect element blocks robustly enough for common COMSOL text meshes.
        if "# number of elements" in low:
            etype = None
            if "tet" in low:
                etype = "tet"
            elif "tri" in low:
                etype = "tri"
            elif "edg" in low or "edge" in low:
                etype = "edge"
            if etype:
                try:
                    n_el = int(line.split("#")[0].strip())
                except Exception:
                    i += 1
                    continue
                for j in range(i + 1, min(i + 10, len(lines))):
                    if "# elements" in lines[j].lower():
                        block = []
                        for k in range(j + 1, min(j + 1 + n_el, len(lines))):
                            vals = lines[k].strip().split()
                            if vals:
                                block.append([int(v) for v in vals])
                        if block:
                            elements[etype] = np.asarray(block, dtype=np.int64)
                        i = j + n_el
                        break
        i += 1

    if not coords:
        raise ValueError("Could not extract mesh coordinates from .mphtxt file.")
    return np.asarray(coords, dtype=np.float32), elements
