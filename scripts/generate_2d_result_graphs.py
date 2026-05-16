#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ai-directional-prediction")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SUMMARY = Path("artifacts/data_experiments/results_2d/summary_2d.csv")
FALLBACK_SUMMARY = Path("artifacts/data_experiments/results_2d/summary.csv")
DEFAULT_MATERIALS = Path("combined_geological_media_parameters.csv")
DEFAULT_CATALOG = Path("backend/data/media/catalog.json")
DEFAULT_FIGURES = Path("figures/results")
DEFAULT_TABLES = Path("tables/results")

MODEL_ORDER = ["pinn", "mgn", "fno", "transformer"]
MODEL_COLORS = {
    "pinn": "#2563EB",
    "mgn": "#059669",
    "fno": "#DC2626",
    "transformer": "#7C3AED",
}
MATERIAL_MARKERS = {
    "basalt": "o",
    "sandstone": "s",
    "granite": "^",
    "limestone": "D",
}
KNOWN_GRAPH_OUTPUTS = [
    "travel_time_by_material_and_model",
    "max_displacement_by_material_and_model",
    "temperature_perturbation_by_material_and_model",
    "directional_response_by_input_azimuth",
    "directional_response_by_azimuth",
    "model_stability_boxplot",
    "vp_vs_travel_time_by_model",
    "density_vs_max_displacement_by_model",
    "young_modulus_vs_max_displacement_by_model",
    "thermal_conductivity_vs_temperature_perturbation_by_model",
    "thermal_expansion_vs_response_magnitude_by_model",
    "physical_parameters_response_correlation_heatmap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate thesis-ready 2D result graphs and summary tables."
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--materials", type=Path, default=DEFAULT_MATERIALS)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--out-figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--out-tables", type=Path, default=DEFAULT_TABLES)
    parser.add_argument(
        "--physics-min-materials",
        type=int,
        default=4,
        help="Minimum number of materials required before physics-based scatter plots are generated.",
    )
    return parser.parse_args()


def normalize_material(value: Any) -> str:
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "sandstone_medium": "sandstone",
        "sandstone_(medium)": "sandstone",
    }
    return aliases.get(text, text)


def normalize_model(value: Any) -> str:
    text = str(value).strip().lower().replace("meshgraphnet", "mgn")
    return text


def load_summary(path: Path) -> tuple[pd.DataFrame, Path, list[str]]:
    warnings: list[str] = []
    actual = path
    if not actual.exists() and path == DEFAULT_SUMMARY and FALLBACK_SUMMARY.exists():
        actual = FALLBACK_SUMMARY
        warnings.append(f"summary_2d.csv was not found; used fallback file {actual}.")
    if not actual.exists():
        raise FileNotFoundError(f"Summary CSV not found: {path}")

    df = pd.read_csv(actual)
    if "material" in df.columns:
        df["material"] = df["material"].map(normalize_material)
    if "model" in df.columns:
        df["model"] = df["model"].map(normalize_model)
    return df, actual, warnings


def load_material_parameters(materials_path: Path, catalog_path: Path) -> tuple[pd.DataFrame, Path, list[str]]:
    warnings: list[str] = []
    if materials_path.exists():
        df = pd.read_csv(materials_path)
        source_path = materials_path
    elif catalog_path.exists():
        df = material_parameters_from_catalog(catalog_path)
        source_path = catalog_path
        warnings.append(
            f"Material parameter CSV {materials_path} was not found; derived parameters from {catalog_path}."
        )
    else:
        raise FileNotFoundError(
            f"Neither material parameter CSV nor media catalog exists: {materials_path}, {catalog_path}"
        )

    if "material" not in df.columns:
        candidate = next((name for name in ("id", "medium_id", "name") if name in df.columns), None)
        if candidate is None:
            raise ValueError("Material table must contain one of: material, id, medium_id, name.")
        df["material"] = df[candidate]
    df["material"] = df["material"].map(normalize_material)
    df = add_physical_helper_columns(df)
    return df, source_path, warnings


def material_parameters_from_catalog(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    rows: list[dict[str, Any]] = []
    for entry in catalog:
        props = entry.get("properties", {})
        rho = float(props.get("rho", np.nan))
        vp = velocity_to_m_s(props.get("vp", np.nan))
        vs = velocity_to_m_s(props.get("vs", np.nan))
        mu = rho * vs * vs if np.isfinite(rho) and np.isfinite(vs) else np.nan
        lambda_ = rho * max(vp * vp - 2.0 * vs * vs, 0.0) if np.isfinite(rho) and np.isfinite(vp) and np.isfinite(vs) else np.nan
        young = (
            mu * (3.0 * lambda_ + 2.0 * mu) / max(lambda_ + mu, 1e-12)
            if np.isfinite(mu) and np.isfinite(lambda_)
            else np.nan
        )
        bulk = lambda_ + 2.0 * mu / 3.0 if np.isfinite(mu) and np.isfinite(lambda_) else np.nan
        rows.append(
            {
                "material": normalize_material(entry.get("id", "")),
                "medium_id": entry.get("id", ""),
                "name": entry.get("name", ""),
                "category": entry.get("category", ""),
                "rho_kg_m3": rho,
                "Vp_m_s": vp,
                "Vs_m_s": vs,
                "E_Pa": young,
                "K_Pa": bulk,
                "mu_Pa": mu,
                "k_W_mK": props.get("thermal_conductivity", np.nan),
                "Cp_J_kgK": props.get("heat_capacity", np.nan),
                "alpha_1_K": props.get("thermal_expansion", np.nan),
                "porosity_percent": float(props.get("porosity_total", np.nan)) * 100.0,
            }
        )
    return pd.DataFrame(rows)


def velocity_to_m_s(value: Any) -> float:
    number = float(value)
    if not np.isfinite(number):
        return np.nan
    return number * 1000.0 if number < 100.0 else number


def add_physical_helper_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for source, target in (("E_Pa", "E_GPa"), ("K_Pa", "K_GPa"), ("mu_Pa", "mu_GPa")):
        if target not in result.columns and source in result.columns:
            result[target] = pd.to_numeric(result[source], errors="coerce") / 1e9
    if "alpha_1e6_K" not in result.columns and "alpha_1_K" in result.columns:
        result["alpha_1e6_K"] = pd.to_numeric(result["alpha_1_K"], errors="coerce") * 1e6
    if "porosity_percent" in result.columns:
        result["porosity_percent"] = result["porosity_percent"].map(parse_porosity_percent)
    return result


def parse_porosity_percent(value: Any) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if not text:
        return np.nan
    token = ""
    for char in text:
        if char.isdigit() or char in ".-":
            token += char
        elif token:
            break
    try:
        return float(token)
    except ValueError:
        return np.nan


def validate_and_filter_2d(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    warnings: list[str] = []
    used = df.copy()
    total_rows = len(used)

    if "status" in used.columns:
        before = len(used)
        used = used[used["status"].astype(str).str.lower() == "ok"].copy()
        if len(used) != before:
            warnings.append(f"Excluded {before - len(used)} rows because status != ok.")
    else:
        warnings.append("Column status is missing; could not filter failed rows.")

    checks = {
        "requested_domain_type": "rect_2d",
        "effective_domain_type": "rect_2d",
    }
    for column, expected in checks.items():
        if column in used.columns:
            before = len(used)
            used = used[used[column].astype(str) == expected].copy()
            if len(used) != before:
                warnings.append(f"Excluded {before - len(used)} rows because {column} != {expected}.")
        else:
            warnings.append(f"Column {column} is missing; 2D domain validation is partial.")

    zero_columns = ["source_z", "probe_z", "direction_z", "elevation_deg"]
    for column in zero_columns:
        if column in used.columns:
            values = pd.to_numeric(used[column], errors="coerce").fillna(0.0)
            before = len(used)
            used = used[values.abs() <= 1e-7].copy()
            if len(used) != before:
                warnings.append(f"Excluded {before - len(used)} rows because {column} is not near zero.")
        else:
            warnings.append(f"Column {column} is missing; geometry validation is partial.")

    meta = {
        "total_rows": total_rows,
        "rows_used": len(used),
        "rows_excluded": total_rows - len(used),
        "all_used_rows_2d": len(used) > 0,
    }
    return used, warnings, meta


def prepare_graph_dataset(summary: pd.DataFrame, materials: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    joined = summary.merge(materials, on="material", how="left", suffixes=("", "_material"))
    missing_materials = sorted(joined.loc[joined["rho_kg_m3"].isna(), "material"].dropna().unique())
    if missing_materials:
        warnings.append(f"Missing material parameters for: {', '.join(missing_materials)}.")
    numeric_columns = [
        "travel_time_ms_pred",
        "max_displacement",
        "max_temperature_perturbation",
        "magnitude",
        "azimuth_deg",
        "rho_kg_m3",
        "Vp_m_s",
        "Vs_m_s",
        "E_GPa",
        "K_GPa",
        "mu_GPa",
        "k_W_mK",
        "Cp_J_kgK",
        "alpha_1e6_K",
        "porosity_percent",
    ]
    for column in numeric_columns:
        if column in joined.columns:
            joined[column] = pd.to_numeric(joined[column], errors="coerce")
    if {"direction_x", "direction_y"}.issubset(joined.columns):
        direction_x = pd.to_numeric(joined["direction_x"], errors="coerce")
        direction_y = pd.to_numeric(joined["direction_y"], errors="coerce")
        joined["input_azimuth_deg"] = np.degrees(np.arctan2(direction_y, direction_x))
    else:
        warnings.append("Columns direction_x/direction_y are missing; input azimuth graph cannot be generated.")
    return joined, warnings


def model_order(values: list[str]) -> list[str]:
    return sorted(values, key=lambda item: MODEL_ORDER.index(item) if item in MODEL_ORDER else len(MODEL_ORDER))


def material_order(values: list[str]) -> list[str]:
    preferred = ["basalt", "sandstone", "granite", "limestone"]
    return sorted(values, key=lambda item: preferred.index(item) if item in preferred else len(preferred))


def ensure_dirs(figures: Path, tables: Path) -> Path:
    figures.mkdir(parents=True, exist_ok=True)
    svg_dir = figures / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return svg_dir


def clean_known_graph_outputs(figures: Path) -> None:
    svg_dir = figures / "svg"
    for stem in KNOWN_GRAPH_OUTPUTS:
        for path in (figures / f"{stem}.png", svg_dir / f"{stem}.svg"):
            if path.exists():
                path.unlink()


def save_figure(fig: plt.Figure, output_path: Path, svg_dir: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    fig.savefig(svg_dir / f"{output_path.stem}.svg")
    plt.close(fig)


def maybe_log_y(ax: plt.Axes, values: pd.Series, warnings: list[str], graph_name: str) -> None:
    positive = pd.to_numeric(values, errors="coerce")
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if len(positive) < 2:
        return
    spread = float(positive.max() / max(float(positive.min()), 1e-18))
    if spread >= 100.0:
        ax.set_yscale("log")
        warnings.append(f"{graph_name}: y-axis uses log scale because values span {spread:.2e}x.")


def scatter_by_model_material(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
    svg_dir: Path,
    warnings: list[str],
) -> bool:
    required = {"model", "material", x, y}
    missing = sorted(required - set(df.columns))
    if missing:
        warnings.append(f"Skipped {output_path.name}: missing columns {missing}.")
        return False
    plot_df = df.dropna(subset=[x, y, "model", "material"])
    if plot_df.empty:
        warnings.append(f"Skipped {output_path.name}: no rows with both {x} and {y}.")
        return False

    fig, ax = plt.subplots(figsize=(9.5, 6))
    for model in model_order(plot_df["model"].dropna().unique().tolist()):
        for material in material_order(plot_df["material"].dropna().unique().tolist()):
            subset = plot_df[(plot_df["model"] == model) & (plot_df["material"] == material)]
            if subset.empty:
                continue
            ax.scatter(
                subset[x],
                subset[y],
                s=52,
                alpha=0.82,
                color=MODEL_COLORS.get(model, "#475569"),
                marker=MATERIAL_MARKERS.get(material, "o"),
                edgecolor="white",
                linewidth=0.5,
                label=f"{model} / {material}",
            )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.28)
    maybe_log_y(ax, plot_df[y], warnings, output_path.name)
    ax.legend(fontsize=8, ncol=2, frameon=True)
    save_figure(fig, output_path, svg_dir)
    return True


def grouped_metric_plot(
    df: pd.DataFrame,
    *,
    metric: str,
    y_label: str,
    title: str,
    output_path: Path,
    svg_dir: Path,
    warnings: list[str],
) -> bool:
    required = {"model", "material", metric}
    missing = sorted(required - set(df.columns))
    if missing:
        warnings.append(f"Skipped {output_path.name}: missing columns {missing}.")
        return False
    plot_df = df.dropna(subset=["model", "material", metric])
    if plot_df.empty:
        warnings.append(f"Skipped {output_path.name}: no values for {metric}.")
        return False

    materials = material_order(plot_df["material"].dropna().unique().tolist())
    models = model_order(plot_df["model"].dropna().unique().tolist())
    grouped = plot_df.groupby(["material", "model"])[metric].agg(["mean", "std"]).reset_index()

    x_positions = np.arange(len(materials))
    width = 0.78 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    for index, model in enumerate(models):
        means: list[float] = []
        errors: list[float] = []
        for material in materials:
            row = grouped[(grouped["material"] == material) & (grouped["model"] == model)]
            means.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
            errors.append(float(row["std"].fillna(0.0).iloc[0]) if not row.empty else 0.0)
        offset = (index - (len(models) - 1) / 2.0) * width
        ax.bar(
            x_positions + offset,
            means,
            width=width,
            yerr=errors,
            capsize=3,
            color=MODEL_COLORS.get(model, "#475569"),
            alpha=0.86,
            label=model,
        )
    ax.set_title(title)
    ax.set_xlabel("Material")
    ax.set_ylabel(y_label)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(materials)
    ax.grid(True, axis="y", alpha=0.28)
    maybe_log_y(ax, plot_df[metric], warnings, output_path.name)
    ax.legend(frameon=True)
    save_figure(fig, output_path, svg_dir)
    return True


def boxplot_metric(
    df: pd.DataFrame,
    *,
    metric: str,
    y_label: str,
    title: str,
    output_path: Path,
    svg_dir: Path,
    warnings: list[str],
) -> bool:
    required = {"model", metric}
    missing = sorted(required - set(df.columns))
    if missing:
        warnings.append(f"Skipped {output_path.name}: missing columns {missing}.")
        return False
    plot_df = df.dropna(subset=["model", metric])
    if plot_df.empty:
        warnings.append(f"Skipped {output_path.name}: no values for {metric}.")
        return False
    models = model_order(plot_df["model"].dropna().unique().tolist())
    data = [plot_df.loc[plot_df["model"] == model, metric].to_numpy() for model in models]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.boxplot(data, tick_labels=models, showfliers=True, patch_artist=True)
    for patch, model in zip(ax.artists, models):
        patch.set_facecolor(MODEL_COLORS.get(model, "#475569"))
        patch.set_alpha(0.75)
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(y_label)
    ax.grid(True, axis="y", alpha=0.28)
    maybe_log_y(ax, plot_df[metric], warnings, output_path.name)
    save_figure(fig, output_path, svg_dir)
    return True


def correlation_heatmap(df: pd.DataFrame, output_path: Path, svg_dir: Path, warnings: list[str]) -> bool:
    columns = [
        "rho_kg_m3",
        "Vp_m_s",
        "Vs_m_s",
        "E_GPa",
        "K_GPa",
        "mu_GPa",
        "k_W_mK",
        "Cp_J_kgK",
        "alpha_1e6_K",
        "porosity_percent",
        "travel_time_ms_pred",
        "max_displacement",
        "max_temperature_perturbation",
        "magnitude",
    ]
    available = [column for column in columns if column in df.columns and df[column].notna().sum() >= 2]
    if len(available) < 3:
        warnings.append(f"Skipped {output_path.name}: not enough numeric columns for correlation heatmap.")
        return False
    corr = df[available].corr(numeric_only=True)
    fig_width = max(9, len(available) * 0.62)
    fig, ax = plt.subplots(figsize=(fig_width, fig_width * 0.82))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Physical Parameters / Output Correlation Heatmap")
    ax.set_xticks(range(len(available)))
    ax.set_yticks(range(len(available)))
    ax.set_xticklabels(available, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(available, fontsize=8)
    for row in range(len(available)):
        for col in range(len(available)):
            value = corr.iloc[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    save_figure(fig, output_path, svg_dir)
    return True


def generate_graphs(
    df: pd.DataFrame,
    figures_dir: Path,
    warnings: list[str],
    *,
    physics_min_materials: int,
) -> list[str]:
    svg_dir = figures_dir / "svg"
    generated: list[str] = []

    core_specs = [
        (
            grouped_metric_plot,
            {
                "metric": "travel_time_ms_pred",
                "y_label": "Predicted travel time (ms)",
                "title": "Travel Time by Material and Model",
                "output_path": figures_dir / "travel_time_by_material_and_model.png",
            },
        ),
        (
            grouped_metric_plot,
            {
                "metric": "max_displacement",
                "y_label": "Maximum displacement",
                "title": "Maximum Displacement by Material and Model",
                "output_path": figures_dir / "max_displacement_by_material_and_model.png",
            },
        ),
        (
            grouped_metric_plot,
            {
                "metric": "max_temperature_perturbation",
                "y_label": "Maximum temperature perturbation",
                "title": "Temperature Perturbation by Material and Model",
                "output_path": figures_dir / "temperature_perturbation_by_material_and_model.png",
            },
        ),
        (
            scatter_by_model_material,
            {
                "x": "input_azimuth_deg",
                "y": "travel_time_ms_pred",
                "x_label": "Input azimuth (deg)",
                "y_label": "Predicted travel time (ms)",
                "title": "Directional Response by Input Azimuth",
                "output_path": figures_dir / "directional_response_by_input_azimuth.png",
            },
        ),
        (
            boxplot_metric,
            {
                "metric": "magnitude",
                "y_label": "Response magnitude",
                "title": "Model Stability / Output Spread",
                "output_path": figures_dir / "model_stability_boxplot.png",
            },
        ),
    ]

    for plotter, kwargs in core_specs:
        if plotter(df, svg_dir=svg_dir, warnings=warnings, **kwargs):
            generated.append(str(kwargs["output_path"]))

    material_count = df["material"].dropna().nunique() if "material" in df.columns else 0
    if material_count < physics_min_materials:
        warnings.append(
            f"Skipped Group B physics-based scatter plots: only {material_count} materials available; "
            f"minimum required is {physics_min_materials}."
        )
        return generated

    physics_specs = [
        (
            scatter_by_model_material,
            {
                "x": "Vp_m_s",
                "y": "travel_time_ms_pred",
                "x_label": "P-wave velocity Vp (m/s)",
                "y_label": "Predicted travel time (ms)",
                "title": "P-wave Velocity vs Predicted Travel Time",
                "output_path": figures_dir / "vp_vs_travel_time_by_model.png",
            },
        ),
        (
            scatter_by_model_material,
            {
                "x": "rho_kg_m3",
                "y": "max_displacement",
                "x_label": "Density (kg/m³)",
                "y_label": "Maximum displacement",
                "title": "Density vs Maximum Displacement",
                "output_path": figures_dir / "density_vs_max_displacement_by_model.png",
            },
        ),
        (
            scatter_by_model_material,
            {
                "x": "E_GPa",
                "y": "max_displacement",
                "x_label": "Young's modulus (GPa)",
                "y_label": "Maximum displacement",
                "title": "Young's Modulus vs Maximum Displacement",
                "output_path": figures_dir / "young_modulus_vs_max_displacement_by_model.png",
            },
        ),
        (
            scatter_by_model_material,
            {
                "x": "k_W_mK",
                "y": "max_temperature_perturbation",
                "x_label": "Thermal conductivity (W/(m·K))",
                "y_label": "Maximum temperature perturbation",
                "title": "Thermal Conductivity vs Temperature Perturbation",
                "output_path": figures_dir / "thermal_conductivity_vs_temperature_perturbation_by_model.png",
            },
        ),
    ]
    for plotter, kwargs in physics_specs:
        if plotter(df, svg_dir=svg_dir, warnings=warnings, **kwargs):
            generated.append(str(kwargs["output_path"]))
    return generated


def write_summary_tables(df: pd.DataFrame, original_summary: pd.DataFrame, tables_dir: Path) -> list[str]:
    generated: list[str] = []
    summary = (
        df.groupby(["model", "material"], dropna=False)
        .agg(
            n_cases=("case_id", "count"),
            mean_travel_time_ms=("travel_time_ms_pred", "mean"),
            std_travel_time_ms=("travel_time_ms_pred", "std"),
            mean_max_displacement=("max_displacement", "mean"),
            std_max_displacement=("max_displacement", "std"),
            mean_max_temperature_perturbation=("max_temperature_perturbation", "mean"),
            std_max_temperature_perturbation=("max_temperature_perturbation", "std"),
            mean_magnitude=("magnitude", "mean"),
            std_magnitude=("magnitude", "std"),
        )
        .reset_index()
    )
    material_summary_csv = tables_dir / "model_material_summary.csv"
    material_summary_tex = tables_dir / "model_material_summary.tex"
    summary.to_csv(material_summary_csv, index=False)
    summary.to_latex(material_summary_tex, index=False, float_format="%.6g")
    generated.extend([str(material_summary_csv), str(material_summary_tex)])

    status_rows: list[dict[str, Any]] = []
    for model in model_order(original_summary["model"].dropna().unique().tolist()):
        subset = original_summary[original_summary["model"] == model]
        status_values = subset["status"].astype(str).str.lower() if "status" in subset.columns else pd.Series([], dtype=str)
        fallback_count = (
            subset["fallback_used"].astype(str).str.lower().eq("true").sum()
            if "fallback_used" in subset.columns
            else "not available"
        )
        effective_domains = (
            ", ".join(sorted(subset["effective_domain_type"].dropna().astype(str).unique()))
            if "effective_domain_type" in subset.columns
            else "not available"
        )
        service_modes = (
            ", ".join(sorted(subset["service_mode"].dropna().astype(str).unique()))
            if "service_mode" in subset.columns
            else "not available"
        )
        notes = f"service_mode={service_modes}"
        status_rows.append(
            {
                "model": model,
                "n_total": len(subset),
                "n_ok": int(status_values.eq("ok").sum()),
                "n_error": int((status_values != "ok").sum()) if len(status_values) else 0,
                "fallback_count": fallback_count,
                "effective_domain_types": effective_domains,
                "notes": notes,
            }
        )
    status = pd.DataFrame(status_rows)
    status_csv = tables_dir / "model_status_summary.csv"
    status_tex = tables_dir / "model_status_summary.tex"
    status.to_csv(status_csv, index=False)
    status.to_latex(status_tex, index=False)
    generated.extend([str(status_csv), str(status_tex)])

    graph_dataset = tables_dir / "graph_dataset_2d.csv"
    df.to_csv(graph_dataset, index=False)
    generated.append(str(graph_dataset))
    return generated


def write_validation_report(
    path: Path,
    *,
    summary_path: Path,
    materials_path: Path,
    validation_meta: dict[str, Any],
    graph_df: pd.DataFrame,
    warnings: list[str],
    generated_graphs: list[str],
    generated_tables: list[str],
) -> None:
    models = model_order(graph_df["model"].dropna().unique().tolist()) if "model" in graph_df.columns else []
    materials = material_order(graph_df["material"].dropna().unique().tolist()) if "material" in graph_df.columns else []
    lines = [
        "2D result graph validation report",
        "=================================",
        "",
        f"Summary input: {summary_path}",
        f"Material parameters input: {materials_path}",
        f"Total rows loaded: {validation_meta['total_rows']}",
        f"Rows used for graphs: {validation_meta['rows_used']}",
        f"Rows excluded: {validation_meta['rows_excluded']}",
        f"Models used: {', '.join(models) if models else 'none'}",
        f"Materials used: {', '.join(materials) if materials else 'none'}",
        f"All used rows are 2D: {validation_meta['all_used_rows_2d']}",
        "",
        "Warnings:",
    ]
    lines.extend([f"- {warning}" for warning in warnings] or ["- none"])
    lines.extend(["", "Generated graphs:"])
    lines.extend([f"- {item}" for item in generated_graphs] or ["- none"])
    lines.extend(["", "Generated tables:"])
    lines.extend([f"- {item}" for item in generated_tables] or ["- none"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_captions(path: Path, generated_graphs: list[str]) -> None:
    captions = {
        "vp_vs_travel_time_by_model.png": (
            "Relationship between P-wave velocity and predicted travel time for the selected 2D source-probe cases.",
            "The plot suggests a qualitative relationship between material wave velocity and model-predicted travel time under controlled 2D comparison conditions.",
        ),
        "density_vs_max_displacement_by_model.png": (
            "Density versus predicted maximum displacement for each model and geological material.",
            "The figure compares mechanical response amplitudes while keeping the interpretation cautious because stiffness and model scaling also affect displacement.",
        ),
        "young_modulus_vs_max_displacement_by_model.png": (
            "Young's modulus versus predicted maximum displacement in the 2D experiment set.",
            "The graph is used to discuss whether stiffer rocks tend to produce smaller predicted deformation under comparable excitation.",
        ),
        "thermal_conductivity_vs_temperature_perturbation_by_model.png": (
            "Thermal conductivity versus predicted maximum temperature perturbation.",
            "The figure supports discussion of how thermal transport parameters relate to local thermal response in the prototype outputs.",
        ),
        "travel_time_by_material_and_model.png": (
            "Mean predicted travel time by material and model with variability across 2D cases.",
            "This figure provides a compact comparison of directional propagation timing across model services and rock presets.",
        ),
        "max_displacement_by_material_and_model.png": (
            "Mean predicted maximum displacement by material and model with variability across 2D cases.",
            "The chart compares mechanical response scales and highlights any model whose output magnitude differs strongly from the others.",
        ),
        "temperature_perturbation_by_material_and_model.png": (
            "Mean predicted maximum temperature perturbation by material and model with variability across 2D cases.",
            "The chart compares thermal response scales and helps identify model outputs that require additional normalization or calibration.",
        ),
        "directional_response_by_input_azimuth.png": (
            "Directional response as a function of the input source direction azimuth in the 2D setup.",
            "The graph uses the same input azimuth definition for every model, avoiding ambiguity from model-specific output direction postprocessing.",
        ),
        "model_stability_boxplot.png": (
            "Distribution of model response magnitude across all selected 2D cases.",
            "The box plot provides a simple view of output spread and relative stability between services.",
        ),
    }
    lines = ["# Figure Captions", ""]
    for graph in generated_graphs:
        name = Path(graph).name
        caption, interpretation = captions.get(
            name,
            (
                f"Generated result graph: {name}.",
                "This figure is interpreted as a comparative prototype result under the simplified 2D experiment setup.",
            ),
        )
        lines.extend(
            [
                f"## `{name}`",
                "",
                f"Suggested LaTeX caption: {caption}",
                "",
                f"Interpretation: {interpretation}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    svg_dir = ensure_dirs(args.out_figures, args.out_tables)
    clean_known_graph_outputs(args.out_figures)
    warnings: list[str] = []

    summary, summary_path, summary_warnings = load_summary(args.summary)
    warnings.extend(summary_warnings)
    materials, material_path, material_warnings = load_material_parameters(args.materials, args.catalog)
    warnings.extend(material_warnings)

    filtered, validation_warnings, validation_meta = validate_and_filter_2d(summary)
    warnings.extend(validation_warnings)
    graph_df, join_warnings = prepare_graph_dataset(filtered, materials)
    warnings.extend(join_warnings)

    generated_graphs = generate_graphs(
        graph_df,
        args.out_figures,
        warnings,
        physics_min_materials=args.physics_min_materials,
    )
    generated_tables = write_summary_tables(graph_df, summary, args.out_tables)
    captions_path = args.out_tables / "figure_captions.md"
    write_captions(captions_path, generated_graphs)
    generated_tables.append(str(captions_path))
    report_path = args.out_tables / "data_validation_report.txt"
    write_validation_report(
        report_path,
        summary_path=summary_path,
        materials_path=material_path,
        validation_meta=validation_meta,
        graph_df=graph_df,
        warnings=warnings,
        generated_graphs=generated_graphs,
        generated_tables=generated_tables,
    )
    generated_tables.append(str(report_path))

    print(
        json.dumps(
            {
                "status": "ok",
                "summary": str(summary_path),
                "materials": str(material_path),
                "rows_loaded": int(validation_meta["total_rows"]),
                "rows_used": int(validation_meta["rows_used"]),
                "models": model_order(graph_df["model"].dropna().unique().tolist()),
                "materials_included": material_order(graph_df["material"].dropna().unique().tolist()),
                "all_used_rows_2d": bool(validation_meta["all_used_rows_2d"]),
                "generated_graphs": generated_graphs,
                "generated_tables": generated_tables,
                "warnings": warnings,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
