#!/usr/bin/env python3
"""Time-series of model predictions vs observation time.

Mirrors the supervisor's Family-1 plots (Green's tensor components vs t)
but reads predictions from the v2 endpoint of each model, so the plots
reflect what the four trained models actually output.

For each thermoelastic-supported material (4 of 10 in catalog_v2.json),
sweep observation.time_s across the training range (4–16 ms with a bit
of padding) and overlay temperature_k, disp_u, disp_v traces from
PINN / FNO / MeshGraphNet / Transformer.

Output: four PNGs into AI_Termoelastic_Waves_Geology/chapters/chapter6/figures/.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-thesis")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests

OUT_DIR = Path(
    "/Users/temporary/unik/AI_Termoelastic_Waves_Geology/chapters/chapter6/figures"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

API = "http://localhost:8000/api/v1/predictions"

MODELS = ["pinn", "fno", "meshgraphnet", "transformer"]
MODEL_LABEL = {
    "pinn": "PINN",
    "fno": "FNO",
    "meshgraphnet": "MeshGraphNet",
    "transformer": "Transformer",
}
MODEL_COLOR = {
    "pinn": "#2563EB",
    "fno": "#DC2626",
    "meshgraphnet": "#059669",
    "transformer": "#9333EA",
}
MATERIALS = [
    ("granite", "Granite"),
    ("sandstone", "Sandstone"),
    ("limestone", "Limestone"),
    ("marble", "Marble"),
]
SOURCE = (0.2, 0.5)
PROBE = (0.8, 0.5)
# observation grid: 50 points across the model training range (4–16 ms)
# plus a small padding on each side.
TIME_GRID_S = np.linspace(0.001, 0.025, 50)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def call(model: str, medium_id: str, time_s: float) -> dict | None:
    body = {
        "schema_version": "2.0",
        "model": model,
        "medium_id": medium_id,
        "geometry": {
            "dimension": 2,
            "source": {"x_m": SOURCE[0], "y_m": SOURCE[1]},
            "probe": {"x_m": PROBE[0], "y_m": PROBE[1]},
        },
        "observation": {"time_s": float(time_s)},
    }
    try:
        r = requests.post(API, json=body, timeout=20)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR {model}/{medium_id}@{time_s}: {e}")
        return None


def _maybe(value):
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def collect(medium_id: str) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for model in MODELS:
        T_arr, U_arr, V_arr = [], [], []
        for t_s in TIME_GRID_S:
            r = call(model, medium_id, t_s)
            if r is None:
                T_arr.append(np.nan); U_arr.append(np.nan); V_arr.append(np.nan)
                continue
            pred = r.get("prediction", {})
            therm = pred.get("thermal", {}).get("temperature_k", {})
            disp = pred.get("displacement", {}).get("components_m", {})
            T_arr.append(_maybe(therm.get("value")))
            U_arr.append(_maybe(disp.get("u")))
            V_arr.append(_maybe(disp.get("v")))
        out[model] = {
            "T": np.array(T_arr, dtype=float),
            "u": np.array(U_arr, dtype=float),
            "v": np.array(V_arr, dtype=float),
        }
    return out


def plot_material(
    mat_label: str, traces: dict[str, dict[str, np.ndarray]], out_path: Path
) -> None:
    """Grid: rows = models, columns = (T, u, v). Each cell has its own
    y-axis scale because the four models output values that span many
    orders of magnitude (calibration differs between routes)."""
    t_ms = TIME_GRID_S * 1000.0
    n_rows = len(MODELS)
    n_cols = 3
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(12.5, 10.5), sharex=True
    )
    fig.suptitle(
        f"Model predictions vs observation time — {mat_label}\n"
        f"source = ({SOURCE[0]:.2f}, {SOURCE[1]:.2f}) m,  "
        f"probe = ({PROBE[0]:.2f}, {PROBE[1]:.2f}) m  ·  "
        f"shaded region = training time range (4–16 ms)",
        fontsize=12,
    )

    col_meta = [
        ("T", "Temperature  $T$  (K)"),
        ("u", "Displacement  $u$  (m)  — x-component"),
        ("v", "Displacement  $v$  (m)  — y-component"),
    ]

    for row, model in enumerate(MODELS):
        for col, (key, ylabel) in enumerate(col_meta):
            ax = axes[row, col]
            y = traces[model][key]
            color = MODEL_COLOR[model]
            if np.all(np.isnan(y)):
                ax.text(
                    0.5, 0.5,
                    f"{MODEL_LABEL[model]}\nno {key} in v2 response",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, alpha=0.55, color=color,
                )
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.plot(
                    t_ms, y, "-", color=color, linewidth=1.4,
                    marker=".", markersize=3, alpha=0.9,
                )
                ax.axvspan(4.0, 16.0, alpha=0.07, color="black", zorder=0)

            if col == 0:
                ax.set_ylabel(MODEL_LABEL[model], fontsize=11, fontweight="bold")
            if row == 0:
                ax.set_title(ylabel, fontsize=10)
            if row == n_rows - 1:
                ax.set_xlabel("Observation time  $t$  (ms)")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    for mat_id, mat_label in MATERIALS:
        print(f"\n=== {mat_label} ({mat_id}) ===")
        t0 = time.perf_counter()
        traces = collect(mat_id)
        elapsed = time.perf_counter() - t0
        print(f"  collected {len(TIME_GRID_S)} x {len(MODELS)} calls in {elapsed:.1f}s")
        out_path = OUT_DIR / f"model_timeseries_{mat_id}.png"
        plot_material(mat_label, traces, out_path)


if __name__ == "__main__":
    main()
