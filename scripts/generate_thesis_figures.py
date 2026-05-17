#!/usr/bin/env python3
"""Generate thesis comparison figures from the 2D 4-material balanced run.

Outputs five PNGs (300 dpi) plus a metrics summary CSV into the thesis
chapter 6 figures folder. Each figure is self-contained and uses a
consistent palette across panels so PINN/FNO/MGN/Transformer are
recognisable across plots.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-thesis")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/Users/temporary/unik/ai-directional-prediction")
SUMMARY = REPO / "artifacts/data_experiments/results_2d_4materials_balanced/summary.csv"
OUT_DIR = Path(
    "/Users/temporary/unik/AI_Termoelastic_Waves_Geology/chapters/chapter6/figures"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
MATERIAL_LABEL = {m: m.capitalize() for m in MATERIALS}

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
        "figure.autolayout": False,
    }
)


def load_summary() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY)
    df["model"] = pd.Categorical(df["model"], categories=MODELS, ordered=True)
    df["material"] = pd.Categorical(df["material"], categories=MATERIALS, ordered=True)
    return df


def _box_by_model_material(
    df: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    fname: str,
    log: bool = False,
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    n_models = len(MODELS)
    n_mats = len(MATERIALS)
    width = 0.18
    positions = np.arange(n_mats)

    for i, model in enumerate(MODELS):
        data = [
            df[(df["model"] == model) & (df["material"] == mat)][column].dropna().values
            for mat in MATERIALS
        ]
        offset = (i - (n_models - 1) / 2) * width
        bp = ax.boxplot(
            data,
            positions=positions + offset,
            widths=width * 0.9,
            patch_artist=True,
            showmeans=False,
            medianprops=dict(color="black", linewidth=1.2),
            flierprops=dict(
                marker="o", markersize=3, markerfacecolor=MODEL_COLOR[model],
                markeredgecolor=MODEL_COLOR[model], alpha=0.7,
            ),
        )
        for box in bp["boxes"]:
            box.set_facecolor(MODEL_COLOR[model])
            box.set_alpha(0.55)
            box.set_edgecolor(MODEL_COLOR[model])
        for whisker in bp["whiskers"] + bp["caps"]:
            whisker.set_color(MODEL_COLOR[model])
            whisker.set_alpha(0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels([MATERIAL_LABEL[m] for m in MATERIALS])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=MODEL_COLOR[m], alpha=0.6, ec=MODEL_COLOR[m])
        for m in MODELS
    ]
    ax.legend(
        legend_handles,
        [MODEL_LABEL[m] for m in MODELS],
        loc="upper right" if not log else "best",
        framealpha=0.9,
        fontsize=9,
    )
    fig.tight_layout()
    out = OUT_DIR / fname
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def plot_travel_time(df: pd.DataFrame) -> None:
    _box_by_model_material(
        df,
        column="travel_time_ms_pred",
        ylabel="Predicted travel time (ms)",
        title="Travel-time prediction by model and material (40 scenarios per cell)",
        fname="travel_time_by_model_material.png",
        log=True,
    )


def plot_displacement(df: pd.DataFrame) -> None:
    _box_by_model_material(
        df,
        column="max_displacement",
        ylabel="Maximum displacement (m)",
        title="Maximum predicted displacement by model and material (log scale)",
        fname="max_displacement_by_model_material.png",
        log=True,
    )


def plot_temperature(df: pd.DataFrame) -> None:
    _box_by_model_material(
        df,
        column="max_temperature_perturbation",
        ylabel="Maximum temperature perturbation (K)",
        title="Maximum predicted temperature perturbation by model and material",
        fname="temperature_perturbation_by_model_material.png",
        log=True,
    )


def plot_model_validity(df: pd.DataFrame) -> None:
    """Heatmap of physical-range validity per (model, metric).

    Each cell shows the share of the 40 scenarios in which the predicted
    value falls into a plausible physical range. This makes calibration
    issues immediately visible.
    """
    valid = pd.DataFrame(index=MODELS, columns=["Travel time", "Max displacement",
                                                "Max ΔT"])
    for model in MODELS:
        sub = df[df["model"] == model]
        n = len(sub)
        if n == 0:
            valid.loc[model] = np.nan
            continue
        valid.loc[model, "Travel time"] = (
            (sub["travel_time_ms_pred"] > 0.05) & (sub["travel_time_ms_pred"] < 5)
        ).mean()
        valid.loc[model, "Max displacement"] = (
            (sub["max_displacement"] > 1e-7) & (sub["max_displacement"] < 1e-2)
        ).mean()
        valid.loc[model, "Max ΔT"] = (
            (sub["max_temperature_perturbation"] > 0.1)
            & (sub["max_temperature_perturbation"] < 100)
        ).mean()

    valid = valid.astype(float)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    im = ax.imshow(valid.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(valid.shape[1]))
    ax.set_xticklabels(valid.columns)
    ax.set_yticks(range(valid.shape[0]))
    ax.set_yticklabels([MODEL_LABEL[m] for m in valid.index])
    ax.grid(False)
    for i in range(valid.shape[0]):
        for j in range(valid.shape[1]):
            v = valid.iat[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                    color="black", fontsize=10, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Fraction within plausible physical range")
    ax.set_title("Predicted value calibration: share of scenarios in physical range")
    fig.tight_layout()
    out = OUT_DIR / "model_validity_matrix.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def plot_dispersion(df: pd.DataFrame) -> None:
    """Per-scenario relative dispersion across the four models.

    For every scenario, compute the ratio of inter-model max/min over the
    four predictions (positive values only). Order-of-magnitude spread
    means the models do not agree on the physical answer — this is the
    direct evidence the user needs in the thesis to qualify the comparison.
    """
    metrics = [
        ("travel_time_ms_pred", "Travel time"),
        ("max_displacement", "Max displacement"),
        ("max_temperature_perturbation", "Max ΔT"),
    ]
    data: dict[str, list[float]] = {}
    for col, label in metrics:
        ratios: list[float] = []
        for case in df["case_id"].unique():
            vals = df[df["case_id"] == case][col].astype(float).values
            vals = vals[vals > 0]
            if len(vals) >= 2:
                ratios.append(vals.max() / vals.min())
        data[label] = ratios

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    positions = np.arange(len(metrics))
    bp = ax.boxplot(
        [data[label] for _, label in metrics],
        positions=positions,
        widths=0.55,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.2),
    )
    palette = ["#3B82F6", "#F59E0B", "#EF4444"]
    for box, color in zip(bp["boxes"], palette):
        box.set_facecolor(color)
        box.set_alpha(0.6)
        box.set_edgecolor(color)
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("Inter-model ratio (max / min)")
    ax.set_title("Cross-model dispersion of predictions per scenario (40 scenarios)")
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.6, linewidth=1)
    ax.text(
        len(metrics) - 0.5,
        1.2,
        "perfect agreement",
        ha="right",
        va="bottom",
        fontsize=8,
        color="black",
    )
    fig.tight_layout()
    out = OUT_DIR / "cross_model_dispersion.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def write_metrics_table(df: pd.DataFrame) -> None:
    """Tabular per-model metric summary (median ± IQR) for inclusion in the thesis."""
    rows = []
    for model in MODELS:
        sub = df[df["model"] == model]
        rows.append(
            {
                "model": MODEL_LABEL[model],
                "n": len(sub),
                "travel_time_median_ms": sub["travel_time_ms_pred"].median(),
                "travel_time_iqr_ms": sub["travel_time_ms_pred"].quantile(0.75)
                - sub["travel_time_ms_pred"].quantile(0.25),
                "max_disp_median_m": sub["max_displacement"].median(),
                "max_disp_iqr_m": sub["max_displacement"].quantile(0.75)
                - sub["max_displacement"].quantile(0.25),
                "max_dT_median_K": sub["max_temperature_perturbation"].median(),
                "max_dT_iqr_K": sub["max_temperature_perturbation"].quantile(0.75)
                - sub["max_temperature_perturbation"].quantile(0.25),
                "fallback_pct": sub["fallback_used"].mean() * 100,
            }
        )
    table = pd.DataFrame(rows)
    out = OUT_DIR / "model_metrics_summary.csv"
    table.to_csv(out, index=False)
    print(f"wrote {out}")
    # Also write a LaTeX-friendly version with scientific notation
    out_tex = OUT_DIR / "model_metrics_summary.tex"
    with open(out_tex, "w") as f:
        f.write(
            "\\begin{tabular}{lrcccc}\n"
            "\\hline\n"
            "Model & n & Travel time (ms) & Max disp.\\ (m) & Max $\\Delta T$ (K) & Fallback \\\\\n"
            "\\hline\n"
        )
        for r in rows:
            f.write(
                f"{r['model']} & {r['n']} & "
                f"{r['travel_time_median_ms']:.3g} $\\pm$ {r['travel_time_iqr_ms']:.2g} & "
                f"{r['max_disp_median_m']:.2g} $\\pm$ {r['max_disp_iqr_m']:.1g} & "
                f"{r['max_dT_median_K']:.2g} $\\pm$ {r['max_dT_iqr_K']:.1g} & "
                f"{r['fallback_pct']:.0f}\\% \\\\\n"
            )
        f.write("\\hline\n\\end{tabular}\n")
    print(f"wrote {out_tex}")


def main() -> None:
    df = load_summary()
    plot_travel_time(df)
    plot_displacement(df)
    plot_temperature(df)
    plot_dispersion(df)
    plot_model_validity(df)
    write_metrics_table(df)


if __name__ == "__main__":
    main()
