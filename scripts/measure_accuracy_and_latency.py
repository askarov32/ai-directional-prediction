#!/usr/bin/env python3
"""Compute travel-time accuracy and per-call latency for the four models.

Accuracy reference: for a P-wave in an isotropic medium, the analytical
travel time between source and probe is

    t_gt = ||x_probe - x_source|| / V_p,

with V_p taken from the medium catalog (km/s converted to m/s). We compare
each model's predicted travel_time_ms_pred against this analytical value.

Latency reference: we re-issue all stored requests through each running
service one by one, measuring per-call wall time around requests.post.
Result is a fresh CSV with per-call latency that supplements the original
summary.csv.

Outputs go to chapters/chapter6/figures/ in the thesis repo.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-thesis")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

REPO = Path("/Users/temporary/unik/ai-directional-prediction")
SUMMARY = REPO / "artifacts/data_experiments/results_2d_4materials_balanced/summary.csv"
REQUESTS_JSONL = (
    REPO
    / "artifacts/data_experiments/results_2d_4materials_balanced/raw/requests.jsonl"
)
CATALOG = REPO / "backend/data/media/catalog.json"
OUT_DIR = Path(
    "/Users/temporary/unik/AI_Termoelastic_Waves_Geology/chapters/chapter6/figures"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SERVICE_URL = {
    "pinn": "http://localhost:9003/predict",
    "mgn": "http://localhost:9001/predict",
    "fno": "http://localhost:9002/predict",
    "transformer": "http://localhost:9004/predict",
}
MODELS = ["pinn", "fno", "mgn", "transformer"]
MODEL_LABEL = {
    "pinn": "PINN",
    "fno": "FNO",
    "mgn": "MeshGraphNet",
    "transformer": "Transformer",
}
MODEL_COLOR = {
    "pinn": "#2563EB",
    "fno": "#DC2626",
    "mgn": "#059669",
    "transformer": "#9333EA",
}
MATERIALS = ["sandstone", "limestone", "basalt", "granite"]

# Map material in summary.csv to medium id in catalog
MATERIAL_TO_CATALOG_ID = {
    "sandstone": "sandstone_medium",
    "limestone": "limestone",
    "basalt": "basalt",
    "granite": "granite",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def load_catalog_vp() -> dict[str, float]:
    cat = json.loads(CATALOG.read_text())
    vp_kms = {}
    for entry in cat:
        vp_kms[entry["id"]] = entry["properties"]["vp"]
    return vp_kms


def compute_analytical_travel_time(df: pd.DataFrame, vp_kms: dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    df["catalog_id"] = df["material"].map(MATERIAL_TO_CATALOG_ID)
    df["vp_kms"] = df["catalog_id"].map(vp_kms)
    df["distance_m"] = np.sqrt(
        (df["probe_x"] - df["source_x"]) ** 2
        + (df["probe_y"] - df["source_y"]) ** 2
    )
    df["travel_time_ms_gt"] = df["distance_m"] / (df["vp_kms"] * 1000.0) * 1000.0
    df["abs_error_ms"] = (df["travel_time_ms_pred"] - df["travel_time_ms_gt"]).abs()
    df["rel_error"] = df["abs_error_ms"] / df["travel_time_ms_gt"]
    return df


def plot_predicted_vs_analytical(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    lo = min(df["travel_time_ms_gt"].min(), df["travel_time_ms_pred"].min()) * 0.5
    hi = max(df["travel_time_ms_gt"].max(), df["travel_time_ms_pred"].max()) * 2.0
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.6, label="y = x (perfect)")
    for model in MODELS:
        sub = df[df["model"] == model]
        ax.scatter(
            sub["travel_time_ms_gt"],
            sub["travel_time_ms_pred"],
            s=28,
            alpha=0.65,
            color=MODEL_COLOR[model],
            label=MODEL_LABEL[model],
            edgecolors="white",
            linewidths=0.4,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"Analytical travel time $\|x_p - x_s\| / V_p$ (ms)")
    ax.set_ylabel("Predicted travel time (ms)")
    ax.set_title("Travel-time accuracy: predicted vs analytical (40 cases × 4 models)")
    ax.legend(loc="upper left", framealpha=0.95, fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    out = OUT_DIR / "travel_time_accuracy_scatter.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def plot_relative_error(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    data = [df[df["model"] == m]["rel_error"].values for m in MODELS]
    bp = ax.boxplot(
        data,
        positions=range(len(MODELS)),
        widths=0.55,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.4),
        flierprops=dict(marker="o", markersize=4, alpha=0.6),
    )
    for box, model in zip(bp["boxes"], MODELS):
        box.set_facecolor(MODEL_COLOR[model])
        box.set_alpha(0.6)
        box.set_edgecolor(MODEL_COLOR[model])
    ax.set_yscale("log")
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS])
    ax.set_ylabel("Relative travel-time error  $|t_\\mathrm{pred}-t_\\mathrm{gt}|/t_\\mathrm{gt}$")
    ax.set_title("Travel-time accuracy by model (lower is better)")
    ax.axhline(0.10, linestyle="--", color="black", alpha=0.4, linewidth=1)
    ax.text(len(MODELS) - 0.5, 0.105, "10% error", ha="right", va="bottom",
            fontsize=8, alpha=0.7)
    fig.tight_layout()
    out = OUT_DIR / "travel_time_relative_error.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def write_accuracy_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        sub = df[df["model"] == model]
        rows.append(
            {
                "model": MODEL_LABEL[model],
                "n": len(sub),
                "mae_ms": sub["abs_error_ms"].mean(),
                "median_abs_error_ms": sub["abs_error_ms"].median(),
                "median_rel_error": sub["rel_error"].median(),
                "p90_rel_error": sub["rel_error"].quantile(0.90),
                "rel_err_lt_10pct": (sub["rel_error"] < 0.10).mean(),
            }
        )
    out_df = pd.DataFrame(rows)
    csv_out = OUT_DIR / "travel_time_accuracy_summary.csv"
    out_df.to_csv(csv_out, index=False)
    print(f"wrote {csv_out}")
    return out_df


def measure_latency(n_reps: int = 20) -> pd.DataFrame:
    """Issue n_reps copies of each unique scenario through every service,
    recording per-call wall time."""
    # Load all stored requests (one per scenario per model, 160 total)
    rows: list[dict] = []
    with open(REQUESTS_JSONL) as f:
        for line in f:
            rows.append(json.loads(line))
    # Take one payload per case_id (the pinn variant; payload is identical across models)
    by_case: dict[str, dict] = {}
    for r in rows:
        if r["model"] == "pinn":
            by_case[r["case_id"]] = r["payload"]
    cases = sorted(by_case)
    # Pick n_reps scenarios per model so per-model totals match
    sample_cases = cases[:n_reps]
    print(f"Measuring latency: {len(sample_cases)} scenarios × {len(MODELS)} models = "
          f"{len(sample_cases) * len(MODELS)} calls")

    records: list[dict] = []
    for model in MODELS:
        url = SERVICE_URL[model]
        for case_id in sample_cases:
            payload = by_case[case_id]
            t0 = time.perf_counter()
            try:
                resp = requests.post(url, json=payload, timeout=30.0)
                latency = time.perf_counter() - t0
                ok = resp.status_code == 200
            except Exception as e:  # noqa: BLE001
                latency = time.perf_counter() - t0
                ok = False
                print(f"  {model}/{case_id} ERROR: {e}")
            records.append(
                {
                    "model": model,
                    "case_id": case_id,
                    "latency_s": latency,
                    "ok": ok,
                }
            )
    df = pd.DataFrame(records)
    csv_out = OUT_DIR / "latency_raw.csv"
    df.to_csv(csv_out, index=False)
    print(f"wrote {csv_out}")
    return df


def _drop_warmup(df: pd.DataFrame) -> pd.DataFrame:
    out_parts = []
    for m, sub in df.groupby("model"):
        out_parts.append(sub.iloc[1:])
    return pd.concat(out_parts, ignore_index=True)


def plot_latency(df: pd.DataFrame) -> None:
    # Drop the first call per model (warm-up) so the boxplot reflects steady state
    df_warm = _drop_warmup(df)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    data = [df_warm[df_warm["model"] == m]["latency_s"].values * 1000 for m in MODELS]
    bp = ax.boxplot(
        data,
        positions=range(len(MODELS)),
        widths=0.55,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.4),
        flierprops=dict(marker="o", markersize=4, alpha=0.6),
    )
    for box, model in zip(bp["boxes"], MODELS):
        box.set_facecolor(MODEL_COLOR[model])
        box.set_alpha(0.6)
        box.set_edgecolor(MODEL_COLOR[model])
    ax.set_yscale("log")
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS])
    ax.set_ylabel("Per-call latency (ms, log)")
    ax.set_title("Inference latency by model (warm steady state, ≥19 calls)")
    fig.tight_layout()
    out = OUT_DIR / "inference_latency_by_model.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def write_combined_summary(acc_df: pd.DataFrame, lat_df: pd.DataFrame) -> None:
    lat_warm = _drop_warmup(lat_df)
    summary = []
    for model in MODELS:
        a = acc_df[acc_df["model"] == MODEL_LABEL[model]].iloc[0]
        l = lat_warm[lat_warm["model"] == model]["latency_s"] * 1000.0
        summary.append(
            {
                "model": MODEL_LABEL[model],
                "n_cases": int(a["n"]),
                "tt_median_rel_error": float(a["median_rel_error"]),
                "tt_rel_err_lt_10pct": float(a["rel_err_lt_10pct"]),
                "latency_median_ms": float(l.median()),
                "latency_p90_ms": float(l.quantile(0.90)),
                "n_latency_samples": int(len(l)),
            }
        )
    out_df = pd.DataFrame(summary)
    csv_out = OUT_DIR / "accuracy_latency_summary.csv"
    out_df.to_csv(csv_out, index=False)
    print(f"wrote {csv_out}")
    # LaTeX table
    tex_out = OUT_DIR / "accuracy_latency_summary.tex"
    with open(tex_out, "w") as f:
        f.write(
            "\\begin{tabular}{lrcccc}\n"
            "\\hline\n"
            "Model & n & Median rel.\\ TT error & TT within 10\\% & "
            "Median latency (ms) & P90 latency (ms) \\\\\n"
            "\\hline\n"
        )
        for r in summary:
            f.write(
                f"{r['model']} & {r['n_cases']} & "
                f"{r['tt_median_rel_error']*100:.1f}\\% & "
                f"{r['tt_rel_err_lt_10pct']*100:.0f}\\% & "
                f"{r['latency_median_ms']:.1f} & "
                f"{r['latency_p90_ms']:.1f} \\\\\n"
            )
        f.write("\\hline\n\\end{tabular}\n")
    print(f"wrote {tex_out}")


def main() -> None:
    vp = load_catalog_vp()
    print("V_p (km/s):", vp)
    df = pd.read_csv(SUMMARY)
    df = compute_analytical_travel_time(df, vp)

    plot_predicted_vs_analytical(df)
    plot_relative_error(df)
    acc_df = write_accuracy_summary(df)

    lat_df = measure_latency(n_reps=20)
    plot_latency(lat_df)

    write_combined_summary(acc_df, lat_df)


if __name__ == "__main__":
    main()
