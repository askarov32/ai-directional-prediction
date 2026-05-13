"""
Minimal synthetic fixture for parser unit tests only.
Do NOT use this file for commercial model training or quality demonstration.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np


def write_fixture(out_dir: str = "tests/fixture_comsol") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    N = 8
    coords = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ], dtype=float) * 0.01
    times = [0.0, 1e-4, 2e-4]

    def write_csv(name, fields):
        header = ["X", "Y", "Z"]
        for f, unit in fields:
            for t in times:
                header.append(f"{f} ({unit}) @ t={t:.1E}")
        rows = ["% Minimal parser fixture", ",".join(header)]
        for i, c in enumerate(coords):
            vals = [f"{c[0]:.6e}", f"{c[1]:.6e}", f"{c[2]:.6e}"]
            for f, unit in fields:
                for ti, t in enumerate(times):
                    if f == "T":
                        val = 293.15 + 10 * ti + i
                    else:
                        val = 1e-9 * (ti + 1) * (i + 1)
                    vals.append(f"{val:.6e}")
            rows.append(",".join(vals))
        (out / name).write_text("\n".join(rows), encoding="utf-8")

    write_csv("data_temperature.csv", [("T", "K")])
    write_csv("data_displacement.csv", [("u", "m"), ("v", "m"), ("w", "m")])
    write_csv("data_stress_1.csv", [("s11", "Pa"), ("s22", "Pa"), ("s33", "Pa"), ("s12", "Pa"), ("s13", "Pa"), ("s23", "Pa")])

    # Very simple mphtxt-like mesh. Parser will extract coords and tet elements.
    lines = [
        "# COMSOL minimal fixture",
        f"{N} # number of mesh points",
        "# Mesh point coordinates",
    ]
    for c in coords:
        lines.append(f"{c[0]} {c[1]} {c[2]}")
    lines += [
        "2 # number of elements tet",
        "# Elements",
        "0 1 2 3",
        "4 5 6 7",
    ]
    (out / "fixture.mphtxt").write_text("\n".join(lines), encoding="utf-8")
    print(f"Fixture written to {out}")


if __name__ == "__main__":
    write_fixture()
