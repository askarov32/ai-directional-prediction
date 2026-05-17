#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_CANDIDATES = [
    ROOT / "artifacts/data_experiments/results_2d_4materials_balanced/summary_2d.csv",
    ROOT / "artifacts/data_experiments/results_2d_4materials/summary_2d.csv",
    ROOT / "artifacts/data_experiments/results_2d/summary_2d.csv",
    ROOT / "artifacts/data_experiments/results_2d_4materials_balanced/summary.csv",
]
DEFAULT_INPUT_CANDIDATES = [
    ROOT / "artifacts/data_experiments/inputs/model_comparison_inputs_2d_4materials_balanced.jsonl",
    ROOT / "artifacts/data_experiments/inputs/model_comparison_inputs_2d_4materials.jsonl",
    ROOT / "artifacts/data_experiments/inputs/model_comparison_inputs_2d.jsonl",
]
DEFAULT_MATERIALS = ROOT / "combined_geological_media_parameters.csv"
DEFAULT_CATALOG = ROOT / "backend/data/media/catalog.json"
DEFAULT_FIGURES = ROOT / "figures/model_comparison"
DEFAULT_TABLES = ROOT / "tables/model_comparison"
DEFAULT_REPORTS_SCAN_ROOTS = [
    ROOT / "artifacts/data_experiments",
    ROOT / "pinn-service/artifacts",
    ROOT / "fno-service/artifacts",
    ROOT / "transformer-service/artifacts",
    ROOT / "data",
    ROOT / "tables",
    ROOT / "figures",
]

MODEL_ORDER = ["pinn", "mgn", "fno", "transformer"]
MODEL_LABELS = {
    "pinn": "PINN",
    "mgn": "MeshGraphNet",
    "fno": "FNO",
    "transformer": "Transformer",
}
MODEL_COLORS = {
    "pinn": "#2563EB",
    "mgn": "#059669",
    "fno": "#DC2626",
    "transformer": "#7C3AED",
}
MATERIAL_ORDER = ["basalt", "sandstone", "granite", "limestone"]
MATERIAL_MARKERS = {
    "basalt": "o",
    "sandstone": "s",
    "granite": "^",
    "limestone": "D",
}
REFERENCE_COLUMN_CANDIDATES = [
    "temperature_true",
    "temperature_target",
    "target_temperature",
    "t_true",
    "t_target",
    "disp_x_true",
    "disp_y_true",
    "disp_z_true",
    "u_true",
    "v_true",
    "w_true",
    "travel_time_true",
    "reference_travel_time_ms",
    "comsol_temperature",
    "comsol_displacement",
]
SPEED_COLUMN_CANDIDATES = [
    "inference_time_ms",
    "latency_ms",
    "prediction_time_ms",
    "duration_ms",
    "elapsed_ms",
    "request_duration_ms",
    "runtime_ms",
]
OUTPUT_METRICS = [
    "travel_time_ms_pred",
    "max_displacement",
    "max_temperature_perturbation",
    "magnitude",
]
FEATURE_COLUMNS = [
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


@dataclass
class Availability:
    summary_path: Path
    material_source_path: Path
    reference_columns: list[str]
    speed_columns: list[str]
    input_files: list[Path]
    summary_files: list[Path]
    material_files: list[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate thesis-ready neural model comparison analysis artifacts.")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--materials", type=Path, default=DEFAULT_MATERIALS)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--out-figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--out-tables", type=Path, default=DEFAULT_TABLES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figures_dir = args.out_figures.expanduser().resolve()
    tables_dir = args.out_tables.expanduser().resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    (figures_dir / "svg").mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary_path = resolve_summary_path(args.summary)
    summary_df = pd.read_csv(summary_path)
    summary_df = normalize_summary(summary_df)

    material_df, material_source_path = load_materials(args.materials, args.catalog)
    availability = collect_availability(
        summary_df=summary_df,
        summary_path=summary_path,
        material_source_path=material_source_path,
    )

    audit_report = build_audit_report(summary_df, material_df, availability)
    write_text(tables_dir / "data_audit_report.md", audit_report)

    validated_df, validation = validate_2d_consistency(summary_df)
    write_text(tables_dir / "data_validation_report.md", build_validation_report(validation))

    joined_df = build_joined_dataset(validated_df, material_df)
    joined_path = tables_dir / "model_comparison_dataset.csv"
    joined_df.to_csv(joined_path, index=False)

    accuracy_report, reference_available = build_accuracy_availability_report(joined_df)
    write_text(tables_dir / "accuracy_availability_report.md", accuracy_report)

    status_df = build_status_summary(summary_df)
    write_csv_and_tex(status_df, tables_dir / "model_status_summary.csv", tables_dir / "model_status_summary.tex")

    stability_df, fno_warnings = build_stability_summary(joined_df)
    write_csv_and_tex(stability_df, tables_dir / "model_stability_summary.csv", tables_dir / "model_stability_summary.tex")

    created_graphs: list[str] = []
    skipped_graphs: list[str] = []
    warnings: list[str] = []

    speed_column = next((col for col in availability.speed_columns if col in joined_df.columns), None)
    if speed_column:
        speed_df = build_speed_summary(joined_df, speed_column)
        write_csv_and_tex(speed_df, tables_dir / "model_speed_summary.csv", tables_dir / "model_speed_summary.tex")
    else:
        skipped_graphs.extend(
            [
                "Graph `inference_time_by_model` skipped because no latency/inference-time column was found in the current summary dataset.",
                "Graph `speed_vs_accuracy_tradeoff` skipped because no latency/inference-time column and no reference target columns were found.",
                "Graph `speed_vs_consistency_tradeoff` skipped because no latency/inference-time column was found in the current summary dataset.",
            ]
        )

    if reference_available:
        skipped_graphs.append("Reference-based accuracy tables/graphs are not implemented in the current workflow.")
    else:
        skipped_graphs.extend(
            [
                "Tables `model_accuracy_summary.csv`, `error_by_material_model.csv`, and `error_by_output_field.csv` were skipped because no explicit ground-truth or target columns were available.",
                "Graphs `error_by_model_and_output_field` and `error_by_material_and_model` were skipped because no explicit ground-truth or target columns were available.",
            ]
        )
        agreement_df, pairwise_df, pinn_df = build_agreement_tables(joined_df)
        agreement_df.to_csv(tables_dir / "model_agreement_summary.csv", index=False)
        pairwise_df.to_csv(tables_dir / "pairwise_model_difference.csv", index=False)
        pinn_df.to_csv(tables_dir / "difference_from_pinn_baseline.csv", index=False)

        agreement_metric = "deviation_from_pinn_score"
        if agreement_metric not in joined_df.columns:
            joined_df = joined_df.merge(
                pinn_df.groupby(["case_id", "model"], as_index=False)[agreement_metric].mean(),
                on=["case_id", "model"],
                how="left",
            )

        create_grouped_material_plot(
            joined_df,
            metric=agreement_metric,
            title="Agreement Deviation by Material and Model",
            y_label="Deviation from PINN baseline (relative score)",
            out_base=figures_dir / "agreement_by_material_and_model",
            created_graphs=created_graphs,
            skipped_graphs=skipped_graphs,
        )
        create_pairwise_difference_heatmap(
            pairwise_df,
            out_base=figures_dir / "pairwise_model_difference_heatmap",
            created_graphs=created_graphs,
            skipped_graphs=skipped_graphs,
        )
        create_feature_scatter(
            joined_df,
            x_col="rho_kg_m3",
            y_col=agreement_metric,
            title="Deviation from PINN Baseline vs Density",
            x_label="Density, kg/m^3",
            y_label="Deviation from PINN baseline (relative score)",
            out_base=figures_dir / "error_or_deviation_vs_density",
            created_graphs=created_graphs,
            skipped_graphs=skipped_graphs,
        )
        create_feature_scatter(
            joined_df,
            x_col="E_GPa",
            y_col=agreement_metric,
            title="Deviation from PINN Baseline vs Young's Modulus",
            x_label="Young's modulus, GPa",
            y_label="Deviation from PINN baseline (relative score)",
            out_base=figures_dir / "error_or_deviation_vs_young_modulus",
            created_graphs=created_graphs,
            skipped_graphs=skipped_graphs,
        )
        create_feature_scatter(
            joined_df,
            x_col="k_W_mK",
            y_col=agreement_metric,
            title="Deviation from PINN Baseline vs Thermal Conductivity",
            x_label="Thermal conductivity, W/(m*K)",
            y_label="Deviation from PINN baseline (relative score)",
            out_base=figures_dir / "error_or_deviation_vs_thermal_conductivity",
            created_graphs=created_graphs,
            skipped_graphs=skipped_graphs,
        )

    directional_ok, directional_warning = verify_input_azimuth_consistency(joined_df)
    if directional_ok:
        create_feature_scatter(
            joined_df,
            x_col="input_azimuth_deg",
            y_col="deviation_from_pinn_score",
            title="Directional Deviation by Input Azimuth",
            x_label="Input azimuth, deg",
            y_label="Deviation from PINN baseline (relative score)",
            out_base=figures_dir / "directional_error_or_deviation_by_azimuth",
            created_graphs=created_graphs,
            skipped_graphs=skipped_graphs,
        )
    else:
        skipped_graphs.append(f"Directional deviation graph skipped because input azimuth is inconsistent: {directional_warning}")

    create_outlier_count_plot(
        stability_df,
        out_base=figures_dir / "outlier_count_by_model",
        created_graphs=created_graphs,
        skipped_graphs=skipped_graphs,
    )

    sensitivity_df = build_feature_association_summary(joined_df)
    sensitivity_df.to_csv(tables_dir / "feature_sensitivity_summary.csv", index=False)
    create_feature_sensitivity_heatmap(
        sensitivity_df,
        out_base=figures_dir / "feature_sensitivity_heatmap",
        created_graphs=created_graphs,
        skipped_graphs=skipped_graphs,
    )

    create_metric_without_fno_plot(
        joined_df,
        metric="max_displacement",
        title="Maximum Displacement by Material and Model (without FNO)",
        y_label="Max displacement",
        out_base=figures_dir / "max_displacement_without_fno",
        created_graphs=created_graphs,
        skipped_graphs=skipped_graphs,
    )
    create_metric_without_fno_plot(
        joined_df,
        metric="max_temperature_perturbation",
        title="Temperature Perturbation by Material and Model (without FNO)",
        y_label="Max temperature perturbation",
        out_base=figures_dir / "temperature_perturbation_without_fno",
        created_graphs=created_graphs,
        skipped_graphs=skipped_graphs,
    )
    create_fno_scale_diagnostic(
        stability_df,
        out_base=figures_dir / "fno_scale_outlier_diagnostic",
        created_graphs=created_graphs,
        skipped_graphs=skipped_graphs,
    )

    write_text(tables_dir / "fno_diagnostic_report.md", build_fno_report(stability_df, fno_warnings))
    write_text(tables_dir / "skipped_graphs_report.md", build_skipped_report(skipped_graphs))
    write_text(
        tables_dir / "figure_captions_and_interpretation.md",
        build_captions_and_interpretation(created_graphs, speed_column is not None, reference_available),
    )
    write_text(
        tables_dir / "final_report.md",
        build_model_comparison_final_report(
            summary_df=summary_df,
            validated_df=validated_df,
            created_graphs=created_graphs,
            skipped_graphs=skipped_graphs,
            material_source_path=material_source_path,
            summary_path=summary_path,
            reference_available=reference_available,
            speed_available=speed_column is not None,
            fno_warnings=fno_warnings,
        ),
    )


def resolve_summary_path(provided: Path | None) -> Path:
    if provided is not None:
        path = provided.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Summary file not found: {path}")
        return path
    for candidate in DEFAULT_SUMMARY_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No default summary file was found.")


def normalize_material(value: Any) -> str:
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "sandstone_medium": "sandstone",
        "sandstone_(medium)": "sandstone",
    }
    return aliases.get(text, text)


def normalize_model(value: Any) -> str:
    text = str(value).strip().lower()
    aliases = {
        "meshgraphnet": "mgn",
        "mesh_graph_net": "mgn",
    }
    return aliases.get(text, text)


def normalize_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "material" in out.columns:
        out["material"] = out["material"].map(normalize_material)
    if "model" in out.columns:
        out["model"] = out["model"].map(normalize_model)
    for column in out.columns:
        if column.endswith("_used"):
            out[column] = out[column].map(_to_bool)
    out["input_azimuth_deg"] = np.degrees(
        np.arctan2(
            pd.to_numeric(out.get("probe_y"), errors="coerce") - pd.to_numeric(out.get("source_y"), errors="coerce"),
            pd.to_numeric(out.get("probe_x"), errors="coerce") - pd.to_numeric(out.get("source_x"), errors="coerce"),
        )
    )
    return out


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def load_materials(materials_path: Path, catalog_path: Path) -> tuple[pd.DataFrame, Path]:
    csv_path = materials_path.expanduser().resolve()
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return normalize_material_table(df), csv_path
    catalog = json.loads(catalog_path.expanduser().resolve().read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for entry in catalog:
        props = entry.get("properties", {})
        rho = float(props.get("rho", np.nan))
        vp = velocity_to_m_s(props.get("vp", np.nan))
        vs = velocity_to_m_s(props.get("vs", np.nan))
        mu = rho * vs * vs if np.isfinite(rho) and np.isfinite(vs) else np.nan
        lambda_ = rho * max(vp * vp - 2.0 * vs * vs, 0.0) if np.isfinite(rho) and np.isfinite(vp) and np.isfinite(vs) else np.nan
        young = mu * (3.0 * lambda_ + 2.0 * mu) / max(lambda_ + mu, 1e-12) if np.isfinite(mu) and np.isfinite(lambda_) else np.nan
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
                "porosity_percent": float(props.get("porosity_total", np.nan)) * 100.0 if props.get("porosity_total") is not None else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    return normalize_material_table(df), catalog_path.expanduser().resolve()


def normalize_material_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    material_col = next((c for c in ["material", "medium_id", "id", "name"] if c in out.columns), None)
    if material_col is None:
        raise ValueError("Material table must include one of: material, medium_id, id, name")
    out["material"] = out[material_col].map(normalize_material)
    if "E_GPa" not in out.columns and "E_Pa" in out.columns:
        out["E_GPa"] = pd.to_numeric(out["E_Pa"], errors="coerce") / 1e9
    if "K_GPa" not in out.columns and "K_Pa" in out.columns:
        out["K_GPa"] = pd.to_numeric(out["K_Pa"], errors="coerce") / 1e9
    if "mu_GPa" not in out.columns and "mu_Pa" in out.columns:
        out["mu_GPa"] = pd.to_numeric(out["mu_Pa"], errors="coerce") / 1e9
    if "alpha_1e6_K" not in out.columns and "alpha_1_K" in out.columns:
        out["alpha_1e6_K"] = pd.to_numeric(out["alpha_1_K"], errors="coerce") * 1e6
    if "porosity_percent" in out.columns:
        out["porosity_percent"] = out["porosity_percent"].map(parse_porosity_percent)
    return out


def parse_porosity_percent(value: Any) -> float:
    if value is None or pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
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


def velocity_to_m_s(value: Any) -> float:
    number = float(value)
    if not np.isfinite(number):
        return np.nan
    return number * 1000.0 if number < 100.0 else number


def collect_availability(summary_df: pd.DataFrame, summary_path: Path, material_source_path: Path) -> Availability:
    input_files = [path for path in DEFAULT_INPUT_CANDIDATES if path.exists()]
    summary_files = [path for path in DEFAULT_SUMMARY_CANDIDATES if path.exists()]
    material_files = [path for path in [DEFAULT_MATERIALS, DEFAULT_CATALOG] if path.exists()]
    reference_columns = [col for col in REFERENCE_COLUMN_CANDIDATES if col in summary_df.columns]
    speed_columns = [col for col in SPEED_COLUMN_CANDIDATES if col in summary_df.columns]
    return Availability(
        summary_path=summary_path,
        material_source_path=material_source_path,
        reference_columns=reference_columns,
        speed_columns=speed_columns,
        input_files=input_files,
        summary_files=summary_files,
        material_files=material_files,
    )


def build_audit_report(summary_df: pd.DataFrame, material_df: pd.DataFrame, availability: Availability) -> str:
    models = sorted(summary_df["model"].dropna().astype(str).unique().tolist(), key=model_sort_key)
    materials = sorted(summary_df["material"].dropna().astype(str).unique().tolist(), key=material_sort_key)
    verifiable_columns = [
        "requested_domain_type",
        "effective_domain_type",
        "source_z",
        "probe_z",
        "direction_z",
        "elevation_deg",
        "domain_lz",
        "domain_nz",
    ]
    present_verifiers = [col for col in verifiable_columns if col in summary_df.columns]
    reference_files = find_reference_like_files()
    lines = [
        "# Data Audit Report",
        "",
        f"- Summary file used: `{availability.summary_path}`",
        f"- Material source used: `{availability.material_source_path}`",
        "",
        "## 1. Input files found",
    ]
    for path in availability.input_files:
        lines.append(f"- `{path}`")
    lines += [
        "",
        "## 2. Summary/result files found",
    ]
    for path in availability.summary_files:
        lines.append(f"- `{path}`")
    lines += [
        "",
        "## 3. Material parameter files found",
    ]
    for path in availability.material_files:
        lines.append(f"- `{path}`")
    lines += [
        "",
        "## 4. Prediction summary columns",
        f"- Column count: `{len(summary_df.columns)}`",
        f"- Columns: `{', '.join(summary_df.columns)}`",
        "",
        "## 5. Material table columns",
        f"- Column count: `{len(material_df.columns)}`",
        f"- Columns: `{', '.join(material_df.columns)}`",
        "",
        "## 6. Models present",
        f"- Models: `{', '.join(models)}`",
        "",
        "## 7. Materials present",
        f"- Materials: `{', '.join(materials)}`",
        "",
        "## 8. 2D consistency verifiability",
        f"- Verifiable columns present: `{', '.join(present_verifiers) if present_verifiers else 'none'}`",
        f"- 2D consistency can be partially verified: `{'yes' if present_verifiers else 'no'}`",
        "",
        "## 9. Speed / latency availability",
        f"- Speed columns found: `{', '.join(availability.speed_columns) if availability.speed_columns else 'none'}`",
        f"- Speed analysis available: `{'yes' if availability.speed_columns else 'no'}`",
        "",
        "## 10. Ground truth / reference availability",
        f"- Reference columns found in summary: `{', '.join(availability.reference_columns) if availability.reference_columns else 'none'}`",
        f"- Reference-like files found: `{', '.join(str(p) for p in reference_files) if reference_files else 'none'}`",
        f"- Accuracy metrics available from current summary: `{'yes' if availability.reference_columns else 'no'}`",
    ]
    return "\n".join(lines) + "\n"


def validate_2d_consistency(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    used = df.copy()
    warnings: list[str] = []
    missing_columns: list[str] = []
    total_rows = len(used)

    if "status" in used.columns:
        used = used[used["status"].astype(str).str.lower() == "ok"].copy()
    else:
        warnings.append("Column status is missing; failed rows cannot be filtered.")

    checks: list[tuple[str, Any, callable[[pd.Series, Any], pd.Series]]] = [
        ("requested_domain_type", "rect_2d", lambda s, v: s.astype(str) == v),
        ("effective_domain_type", "rect_2d", lambda s, v: s.astype(str) == v),
        ("source_z", 0.0, lambda s, v: pd.to_numeric(s, errors="coerce").fillna(np.nan).abs() <= 1e-7),
        ("probe_z", 0.0, lambda s, v: pd.to_numeric(s, errors="coerce").fillna(np.nan).abs() <= 1e-7),
        ("direction_z", 0.0, lambda s, v: pd.to_numeric(s, errors="coerce").fillna(np.nan).abs() <= 1e-7),
        ("elevation_deg", 0.0, lambda s, v: pd.to_numeric(s, errors="coerce").fillna(np.nan).abs() <= 1e-7),
        ("domain_lz", 0.0, lambda s, v: pd.to_numeric(s, errors="coerce").fillna(np.nan).abs() <= 1e-7),
        ("domain_nz", 1, lambda s, v: pd.to_numeric(s, errors="coerce") == v),
    ]

    for column, expected, predicate in checks:
        if column not in used.columns:
            missing_columns.append(column)
            continue
        mask = predicate(used[column], expected)
        before = len(used)
        used = used[mask.fillna(False)].copy()
        if len(used) != before:
            warnings.append(f"Excluded {before - len(used)} rows because `{column}` is inconsistent with the strict 2D expectation.")

    numeric_checks = ["travel_time_ms_pred", "max_displacement", "max_temperature_perturbation", "magnitude"]
    non_finite_by_column: dict[str, int] = {}
    for column in numeric_checks:
        if column in used.columns:
            values = pd.to_numeric(used[column], errors="coerce")
            count = int((~np.isfinite(values)).sum())
            non_finite_by_column[column] = count

    validation = {
        "total_rows": total_rows,
        "valid_rows": len(used),
        "excluded_rows": total_rows - len(used),
        "models": sorted(used["model"].dropna().astype(str).unique().tolist(), key=model_sort_key) if "model" in used.columns else [],
        "materials": sorted(used["material"].dropna().astype(str).unique().tolist(), key=material_sort_key) if "material" in used.columns else [],
        "status_counts": df["status"].astype(str).value_counts(dropna=False).to_dict() if "status" in df.columns else {},
        "fallback_counts": df["fallback_used"].map(_to_bool).value_counts(dropna=False).to_dict() if "fallback_used" in df.columns else {},
        "non_finite_by_column": non_finite_by_column,
        "missing_columns": missing_columns,
        "warnings": warnings,
    }
    return used, validation


def build_validation_report(validation: dict[str, Any]) -> str:
    lines = [
        "# Data Validation Report",
        "",
        f"- Total rows: `{validation['total_rows']}`",
        f"- Valid 2D rows: `{validation['valid_rows']}`",
        f"- Excluded rows: `{validation['excluded_rows']}`",
        f"- Models: `{', '.join(validation['models']) if validation['models'] else 'none'}`",
        f"- Materials: `{', '.join(validation['materials']) if validation['materials'] else 'none'}`",
        f"- Status counts: `{validation['status_counts']}`",
        f"- Fallback counts: `{validation['fallback_counts']}`",
        f"- Non-finite values: `{validation['non_finite_by_column']}`",
        f"- Missing columns: `{', '.join(validation['missing_columns']) if validation['missing_columns'] else 'none'}`",
        "",
        "## Warnings",
    ]
    if validation["warnings"]:
        lines.extend(f"- {warning}" for warning in validation["warnings"])
    else:
        lines.append("- No 2D consistency warnings.")
    return "\n".join(lines) + "\n"


def build_joined_dataset(summary_df: pd.DataFrame, material_df: pd.DataFrame) -> pd.DataFrame:
    joined = summary_df.merge(material_df, on="material", how="left", suffixes=("", "_material"))
    for column in OUTPUT_METRICS + ["azimuth_deg", "input_azimuth_deg", "rho_kg_m3", "Vp_m_s", "Vs_m_s", "E_GPa", "K_GPa", "mu_GPa", "k_W_mK", "Cp_J_kgK", "alpha_1e6_K", "porosity_percent"]:
        if column in joined.columns:
            joined[column] = pd.to_numeric(joined[column], errors="coerce")
    return joined


def build_accuracy_availability_report(df: pd.DataFrame) -> tuple[str, bool]:
    reference_columns = [column for column in REFERENCE_COLUMN_CANDIDATES if column in df.columns and df[column].notna().any()]
    available = bool(reference_columns)
    lines = [
        "# Accuracy Availability Report",
        "",
        f"- Accuracy metrics available: `{'yes' if available else 'no'}`",
        f"- Reference source used: `{', '.join(reference_columns) if reference_columns else 'none found in current summary/joined dataset'}`",
    ]
    if not available:
        lines += [
            "",
            "The current analysis dataset does not contain explicit ground-truth or reference target columns.",
            "Therefore, the generated thesis artifacts treat the comparison as model agreement, physical consistency, stability, and observed feature-output association analysis rather than accuracy evaluation.",
        ]
    return "\n".join(lines) + "\n", available


def build_status_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, model_df in df.groupby("model"):
        total = len(model_df)
        ok = int((model_df["status"].astype(str).str.lower() == "ok").sum()) if "status" in model_df.columns else total
        fallback = int(model_df["fallback_used"].map(_to_bool).sum()) if "fallback_used" in model_df.columns else 0
        error = int((model_df["status"].astype(str).str.lower() == "error").sum()) if "status" in model_df.columns else 0
        timeout = int((model_df.get("error_code", pd.Series(dtype=str)).astype(str).str.contains("TIMEOUT", case=False, na=False)).sum())
        rows.append(
            {
                "model": model,
                "total_rows": total,
                "ok_rows": ok,
                "failed_rows": total - ok,
                "fallback_count": fallback,
                "error_rows": error,
                "timeout_rows": timeout,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("model", key=lambda s: s.map(model_sort_key))


def build_stability_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for model, model_df in df.groupby("model"):
        non_finite = 0
        for column in OUTPUT_METRICS + ["azimuth_deg"]:
            if column in model_df.columns:
                values = pd.to_numeric(model_df[column], errors="coerce")
                non_finite += int((~np.isfinite(values)).sum())
        outlier_displacement = int((pd.to_numeric(model_df.get("max_displacement"), errors="coerce").abs() > 1e2).fillna(False).sum())
        outlier_temperature = int((pd.to_numeric(model_df.get("max_temperature_perturbation"), errors="coerce").abs() > 1e4).fillna(False).sum())
        outlier_magnitude = int((pd.to_numeric(model_df.get("magnitude"), errors="coerce") > 1e6).fillna(False).sum())
        fallback_count = int(model_df.get("fallback_used", pd.Series(dtype=bool)).map(_to_bool).sum())
        composite_outlier = outlier_displacement + outlier_temperature + outlier_magnitude
        row = {
            "model": model,
            "rows": len(model_df),
            "ok_rows": int((model_df["status"].astype(str).str.lower() == "ok").sum()) if "status" in model_df.columns else len(model_df),
            "failed_rows": int((model_df["status"].astype(str).str.lower() == "error").sum()) if "status" in model_df.columns else 0,
            "fallback_count": fallback_count,
            "non_finite_output_count": non_finite,
            "scale_outlier_count": composite_outlier,
            "max_displacement_mean": safe_mean(model_df.get("max_displacement")),
            "max_displacement_std": safe_std(model_df.get("max_displacement")),
            "max_temperature_perturbation_mean": safe_mean(model_df.get("max_temperature_perturbation")),
            "max_temperature_perturbation_std": safe_std(model_df.get("max_temperature_perturbation")),
            "travel_time_ms_pred_mean": safe_mean(model_df.get("travel_time_ms_pred")),
            "travel_time_ms_pred_std": safe_std(model_df.get("travel_time_ms_pred")),
            "magnitude_mean": safe_mean(model_df.get("magnitude")),
            "magnitude_std": safe_std(model_df.get("magnitude")),
            "displacement_cv": safe_cv(model_df.get("max_displacement")),
            "temperature_cv": safe_cv(model_df.get("max_temperature_perturbation")),
        }
        rows.append(row)
        if model == "fno" and composite_outlier > 0:
            warnings.append(
                "FNO produces scale outliers relative to the other models; treat these results as scale-unstable prototype outputs rather than validated physical displacements or temperatures."
            )
    out = pd.DataFrame(rows)
    out = out.sort_values("model", key=lambda s: s.map(model_sort_key))
    return out, warnings


def safe_mean(series: pd.Series | Any) -> float:
    if series is None:
        return np.nan
    values = pd.to_numeric(series, errors="coerce")
    return float(values.mean()) if len(values) else np.nan


def safe_std(series: pd.Series | Any) -> float:
    if series is None:
        return np.nan
    values = pd.to_numeric(series, errors="coerce")
    return float(values.std()) if len(values) else np.nan


def safe_cv(series: pd.Series | Any) -> float:
    mean = safe_mean(series)
    std = safe_std(series)
    if not np.isfinite(mean) or abs(mean) < 1e-12 or not np.isfinite(std):
        return np.nan
    return float(std / abs(mean))


def build_speed_summary(df: pd.DataFrame, speed_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, model_df in df.groupby("model"):
        values = pd.to_numeric(model_df[speed_column], errors="coerce").dropna()
        rows.append(
            {
                "model": model,
                "speed_column": speed_column,
                "mean_inference_time_ms": float(values.mean()) if len(values) else np.nan,
                "std_inference_time_ms": float(values.std()) if len(values) else np.nan,
                "median_inference_time_ms": float(values.median()) if len(values) else np.nan,
                "min_inference_time_ms": float(values.min()) if len(values) else np.nan,
                "max_inference_time_ms": float(values.max()) if len(values) else np.nan,
                "p95_inference_time_ms": float(np.percentile(values, 95)) if len(values) >= 5 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("model", key=lambda s: s.map(model_sort_key))


def build_agreement_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    grouped_metrics = work.groupby("case_id")[OUTPUT_METRICS].median().rename(columns=lambda c: f"{c}_ensemble_median")
    work = work.merge(grouped_metrics, on="case_id", how="left")

    pinn_baseline = (
        work[work["model"] == "pinn"][["case_id"] + OUTPUT_METRICS]
        .rename(columns={metric: f"{metric}_pinn" for metric in OUTPUT_METRICS})
        .drop_duplicates(subset=["case_id"])
    )
    work = work.merge(pinn_baseline, on="case_id", how="left")

    difference_rows: list[dict[str, Any]] = []
    for metric in OUTPUT_METRICS:
        median_col = f"{metric}_ensemble_median"
        pinn_col = f"{metric}_pinn"
        work[f"{metric}_abs_diff_median"] = (work[metric] - work[median_col]).abs()
        work[f"{metric}_rel_diff_median"] = work[f"{metric}_abs_diff_median"] / (work[median_col].abs() + 1e-12)
        work[f"{metric}_abs_diff_pinn"] = (work[metric] - work[pinn_col]).abs()
        work[f"{metric}_rel_diff_pinn"] = work[f"{metric}_abs_diff_pinn"] / (work[pinn_col].abs() + 1e-12)
        difference_rows.append(
            {
                "metric": metric,
                "mean_abs_diff_from_ensemble_median": float(work[f"{metric}_abs_diff_median"].mean()),
                "mean_rel_diff_from_ensemble_median": float(work[f"{metric}_rel_diff_median"].mean()),
                "mean_abs_diff_from_pinn": float(work[f"{metric}_abs_diff_pinn"].mean()),
                "mean_rel_diff_from_pinn": float(work[f"{metric}_rel_diff_pinn"].mean()),
            }
        )

    work["deviation_from_pinn_score"] = work[[f"{metric}_rel_diff_pinn" for metric in OUTPUT_METRICS]].mean(axis=1)
    work["deviation_from_ensemble_score"] = work[[f"{metric}_rel_diff_median" for metric in OUTPUT_METRICS]].mean(axis=1)

    agreement_summary = (
        work.groupby("model", as_index=False)
        .agg(
            rows=("case_id", "count"),
            mean_deviation_from_pinn=("deviation_from_pinn_score", "mean"),
            std_deviation_from_pinn=("deviation_from_pinn_score", "std"),
            mean_deviation_from_ensemble=("deviation_from_ensemble_score", "mean"),
            std_deviation_from_ensemble=("deviation_from_ensemble_score", "std"),
        )
        .sort_values("model", key=lambda s: s.map(model_sort_key))
    )

    pairwise_rows: list[dict[str, Any]] = []
    for case_id, case_df in work.groupby("case_id"):
        case_df = case_df.sort_values("model", key=lambda s: s.map(model_sort_key))
        records = case_df.to_dict(orient="records")
        for index, left in enumerate(records):
            for right in records[index + 1 :]:
                row = {"case_id": case_id, "model_a": left["model"], "model_b": right["model"]}
                for metric in OUTPUT_METRICS:
                    left_value = left.get(metric)
                    right_value = right.get(metric)
                    row[f"{metric}_abs_diff"] = abs(left_value - right_value) if pd.notna(left_value) and pd.notna(right_value) else np.nan
                    row[f"{metric}_rel_diff"] = row[f"{metric}_abs_diff"] / (abs(right_value) + 1e-12) if pd.notna(row[f"{metric}_abs_diff"]) else np.nan
                pairwise_rows.append(row)
    pairwise_df = pd.DataFrame(pairwise_rows)

    pinn_df = work[
        [
            "case_id",
            "model",
            "material",
            "deviation_from_pinn_score",
            "deviation_from_ensemble_score",
        ]
        + [f"{metric}_abs_diff_pinn" for metric in OUTPUT_METRICS]
        + [f"{metric}_rel_diff_pinn" for metric in OUTPUT_METRICS]
    ].copy()

    return agreement_summary, pairwise_df, pinn_df


def verify_input_azimuth_consistency(df: pd.DataFrame) -> tuple[bool, str]:
    if "input_azimuth_deg" not in df.columns:
        return False, "column input_azimuth_deg is missing"
    for case_id, case_df in df.groupby("case_id"):
        values = pd.to_numeric(case_df["input_azimuth_deg"], errors="coerce").dropna().round(8).unique().tolist()
        if len(values) > 1:
            return False, f"case_id {case_id} has model-dependent input azimuth values {values}"
    return True, ""


def build_feature_association_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, model_df in df.groupby("model"):
        for feature in FEATURE_COLUMNS:
            if feature not in model_df.columns:
                continue
            output_scores = []
            for metric in OUTPUT_METRICS:
                if metric not in model_df.columns:
                    continue
                pair = model_df[[feature, metric]].dropna()
                if len(pair) < 3 or pair[feature].nunique() < 2 or pair[metric].nunique() < 2:
                    continue
                corr = pair[feature].rank().corr(pair[metric].rank())
                if pd.notna(corr):
                    output_scores.append(abs(float(corr)))
            if output_scores:
                rows.append(
                    {
                        "model": model,
                        "feature": feature,
                        "association_type": "observed_spearman_abs_correlation",
                        "mean_observed_association": float(np.mean(output_scores)),
                        "max_observed_association": float(np.max(output_scores)),
                        "n_outputs_used": len(output_scores),
                    }
                )
    return pd.DataFrame(rows).sort_values(["feature", "model"], key=lambda s: s.map(model_sort_key) if s.name == "model" else s)


def create_grouped_material_plot(
    df: pd.DataFrame,
    *,
    metric: str,
    title: str,
    y_label: str,
    out_base: Path,
    created_graphs: list[str],
    skipped_graphs: list[str],
) -> None:
    required = {"material", "model", metric}
    if not required.issubset(df.columns):
        skipped_graphs.append(f"{out_base.name} skipped because required columns are missing.")
        return
    plot_df = df.dropna(subset=["material", "model", metric])
    if plot_df.empty:
        skipped_graphs.append(f"{out_base.name} skipped because there are no usable rows.")
        return
    grouped = plot_df.groupby(["material", "model"])[metric].agg(["mean", "std"]).reset_index()
    materials = sorted(grouped["material"].unique().tolist(), key=material_sort_key)
    models = sorted(grouped["model"].unique().tolist(), key=model_sort_key)
    x = np.arange(len(materials))
    width = 0.78 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, model in enumerate(models):
        subset = grouped[grouped["model"] == model]
        means = [float(subset.loc[subset["material"] == material, "mean"].iloc[0]) if not subset.loc[subset["material"] == material].empty else np.nan for material in materials]
        stds = [float(subset.loc[subset["material"] == material, "std"].fillna(0.0).iloc[0]) if not subset.loc[subset["material"] == material].empty else 0.0 for material in materials]
        offset = (idx - (len(models) - 1) / 2.0) * width
        ax.bar(x + offset, means, width=width, yerr=stds, capsize=3, color=MODEL_COLORS.get(model, "#64748B"), alpha=0.86, label=MODEL_LABELS.get(model, model))
    ax.set_title(title)
    ax.set_xlabel("Material")
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(materials)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=True)
    save_plot(fig, out_base, created_graphs)


def create_pairwise_difference_heatmap(
    pairwise_df: pd.DataFrame,
    *,
    out_base: Path,
    created_graphs: list[str],
    skipped_graphs: list[str],
) -> None:
    metric = "max_displacement_abs_diff"
    if pairwise_df.empty or metric not in pairwise_df.columns:
        skipped_graphs.append(f"{out_base.name} skipped because pairwise difference data is unavailable.")
        return
    models = MODEL_ORDER
    matrix = pd.DataFrame(np.nan, index=models, columns=models)
    for model_a in models:
        for model_b in models:
            if model_a == model_b:
                matrix.loc[model_a, model_b] = 0.0
                continue
            subset = pairwise_df[
                ((pairwise_df["model_a"] == model_a) & (pairwise_df["model_b"] == model_b))
                | ((pairwise_df["model_a"] == model_b) & (pairwise_df["model_b"] == model_a))
            ]
            matrix.loc[model_a, model_b] = subset[metric].mean()
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(np.log10(matrix.astype(float).fillna(1e-12).to_numpy() + 1e-12), cmap="magma")
    ax.set_title("Pairwise Model Difference Heatmap\n(log10 mean absolute displacement difference)")
    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABELS.get(model, model) for model in models], rotation=30, ha="right")
    ax.set_yticklabels([MODEL_LABELS.get(model, model) for model in models])
    for row in range(len(models)):
        for col in range(len(models)):
            value = matrix.iloc[row, col]
            text = "0" if row == col else f"{value:.2e}" if pd.notna(value) else "nan"
            ax.text(col, row, text, ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(image, ax=ax, label="log10 difference")
    save_plot(fig, out_base, created_graphs)


def create_feature_scatter(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    out_base: Path,
    created_graphs: list[str],
    skipped_graphs: list[str],
) -> None:
    required = {"model", "material", x_col, y_col}
    if not required.issubset(df.columns):
        skipped_graphs.append(f"{out_base.name} skipped because required columns are missing.")
        return
    plot_df = df.dropna(subset=["model", "material", x_col, y_col])
    if plot_df.empty:
        skipped_graphs.append(f"{out_base.name} skipped because there are no usable rows.")
        return
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for model in sorted(plot_df["model"].unique().tolist(), key=model_sort_key):
        for material in sorted(plot_df["material"].unique().tolist(), key=material_sort_key):
            subset = plot_df[(plot_df["model"] == model) & (plot_df["material"] == material)]
            if subset.empty:
                continue
            ax.scatter(
                subset[x_col],
                subset[y_col],
                color=MODEL_COLORS.get(model, "#64748B"),
                marker=MATERIAL_MARKERS.get(material, "o"),
                edgecolor="white",
                linewidth=0.5,
                s=52,
                alpha=0.82,
                label=f"{MODEL_LABELS.get(model, model)} / {material}",
            )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2, frameon=True)
    save_plot(fig, out_base, created_graphs)


def create_outlier_count_plot(
    stability_df: pd.DataFrame,
    *,
    out_base: Path,
    created_graphs: list[str],
    skipped_graphs: list[str],
) -> None:
    if stability_df.empty or "scale_outlier_count" not in stability_df.columns:
        skipped_graphs.append(f"{out_base.name} skipped because stability summary is unavailable.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ordered = stability_df.sort_values("model", key=lambda s: s.map(model_sort_key))
    labels = [MODEL_LABELS.get(model, model) for model in ordered["model"]]
    values = ordered["scale_outlier_count"].tolist()
    ax.bar(labels, values, color=[MODEL_COLORS.get(model, "#64748B") for model in ordered["model"]], alpha=0.86)
    ax.set_title("Outlier Count by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Number of prototype outlier warnings")
    ax.grid(True, axis="y", alpha=0.25)
    save_plot(fig, out_base, created_graphs)


def create_feature_sensitivity_heatmap(
    sensitivity_df: pd.DataFrame,
    *,
    out_base: Path,
    created_graphs: list[str],
    skipped_graphs: list[str],
) -> None:
    if sensitivity_df.empty:
        skipped_graphs.append(f"{out_base.name} skipped because no feature association data could be computed.")
        return
    features = [feature for feature in FEATURE_COLUMNS if feature in sensitivity_df["feature"].unique().tolist()]
    models = MODEL_ORDER
    matrix = pd.DataFrame(np.nan, index=features, columns=models)
    for _, row in sensitivity_df.iterrows():
        matrix.loc[row["feature"], row["model"]] = row["mean_observed_association"]
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    image = ax.imshow(matrix.fillna(0.0).to_numpy(), cmap="viridis", vmin=0.0, vmax=max(0.3, float(np.nanmax(matrix.to_numpy())) if np.isfinite(np.nanmax(matrix.to_numpy())) else 0.3))
    ax.set_title("Observed Feature-Output Association Heatmap")
    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(features)))
    ax.set_xticklabels([MODEL_LABELS.get(model, model) for model in models], rotation=30, ha="right")
    ax.set_yticklabels(features)
    for row in range(len(features)):
        for col in range(len(models)):
            value = matrix.iloc[row, col]
            text = f"{value:.2f}" if pd.notna(value) else "NA"
            ax.text(col, row, text, ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(image, ax=ax, label="Mean absolute observed association")
    save_plot(fig, out_base, created_graphs)


def create_metric_without_fno_plot(
    df: pd.DataFrame,
    *,
    metric: str,
    title: str,
    y_label: str,
    out_base: Path,
    created_graphs: list[str],
    skipped_graphs: list[str],
) -> None:
    subset = df[df["model"] != "fno"].copy()
    if subset.empty or metric not in subset.columns:
        skipped_graphs.append(f"{out_base.name} skipped because non-FNO subset is unavailable.")
        return
    create_grouped_material_plot(
        subset,
        metric=metric,
        title=title,
        y_label=y_label,
        out_base=out_base,
        created_graphs=created_graphs,
        skipped_graphs=skipped_graphs,
    )


def create_fno_scale_diagnostic(
    stability_df: pd.DataFrame,
    *,
    out_base: Path,
    created_graphs: list[str],
    skipped_graphs: list[str],
) -> None:
    if stability_df.empty:
        skipped_graphs.append(f"{out_base.name} skipped because stability data is unavailable.")
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ordered = stability_df.sort_values("model", key=lambda s: s.map(model_sort_key))
    labels = [MODEL_LABELS.get(model, model) for model in ordered["model"]]
    displacement = ordered["max_displacement_mean"].astype(float).replace(0.0, np.nan)
    ax.bar(labels, displacement, color=[MODEL_COLORS.get(model, "#64748B") for model in ordered["model"]], alpha=0.86)
    ax.set_yscale("log")
    ax.set_title("Model Scale Diagnostic: Mean Max Displacement")
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean max displacement (log scale)")
    ax.grid(True, axis="y", alpha=0.25)
    save_plot(fig, out_base, created_graphs)


def save_plot(fig: plt.Figure, out_base: Path, created_graphs: list[str]) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    svg_dir = out_base.parent / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    png_path = out_base.with_suffix(".png")
    svg_path = svg_dir / f"{out_base.name}.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)
    created_graphs.append(str(png_path))
    created_graphs.append(str(svg_path))


def build_fno_report(stability_df: pd.DataFrame, warnings: list[str]) -> str:
    row = stability_df[stability_df["model"] == "fno"]
    lines = ["# FNO Diagnostic Report", ""]
    if row.empty:
        lines.append("No FNO rows were available in the validated comparison dataset.")
    else:
        record = row.iloc[0].to_dict()
        lines.append(f"- Rows: `{record['rows']}`")
        lines.append(f"- Fallback count: `{record['fallback_count']}`")
        lines.append(f"- Non-finite output count: `{record['non_finite_output_count']}`")
        lines.append(f"- Scale outlier count: `{record['scale_outlier_count']}`")
        lines.append(f"- Mean max displacement: `{record['max_displacement_mean']}`")
        lines.append(f"- Mean max temperature perturbation: `{record['max_temperature_perturbation_mean']}`")
        lines.append("")
        lines.append("Interpretation:")
        lines.append("FNO output values are treated as scale-unstable prototype predictions when they exceed the range of other models by several orders of magnitude.")
    if warnings:
        lines += ["", "Warnings:"] + [f"- {warning}" for warning in warnings]
    return "\n".join(lines) + "\n"


def build_skipped_report(skipped_graphs: list[str]) -> str:
    lines = ["# Skipped Graphs Report", ""]
    if skipped_graphs:
        lines.extend(f"- {item}" for item in skipped_graphs)
    else:
        lines.append("- No graphs were skipped.")
    return "\n".join(lines) + "\n"


def build_captions_and_interpretation(created_graphs: list[str], speed_available: bool, reference_available: bool) -> str:
    unique_pngs = [path for path in created_graphs if path.endswith(".png")]
    captions: dict[str, tuple[str, str, str]] = {
        "agreement_by_material_and_model.png": (
            "Agreement Deviation by Material and Model",
            "The plot compares how far each model deviates from the physics-informed PINN baseline under identical 2D source-probe conditions for each rock type.",
            "These deviations indicate comparative prototype behavior rather than physical error against ground truth.",
        ),
        "pairwise_model_difference_heatmap.png": (
            "Pairwise Model Difference Heatmap",
            "The heatmap summarizes pairwise differences in predicted maximum displacement between model services across the analyzed 2D cases.",
            "Large values, especially when dominated by FNO, should be interpreted as scale instability or model disagreement rather than proof of real physical divergence.",
        ),
        "error_or_deviation_vs_density.png": (
            "Deviation vs Density",
            "The plot shows how the model deviation score changes with material density across the available 2D experiments.",
            "The trend suggests an observed association only; it should not be interpreted as field-validated proof of density-controlled wave behavior.",
        ),
        "error_or_deviation_vs_young_modulus.png": (
            "Deviation vs Young's Modulus",
            "The plot compares deviation from the PINN baseline as a function of estimated Young's modulus for the analyzed rocks.",
            "The relationship is qualitative and reflects comparative model behavior within the prototype setup.",
        ),
        "error_or_deviation_vs_thermal_conductivity.png": (
            "Deviation vs Thermal Conductivity",
            "The plot compares deviation from the PINN baseline against thermal conductivity for the available geological media.",
            "This should be treated as a prototype-level observed association, not a validated thermoelastic law.",
        ),
        "directional_error_or_deviation_by_azimuth.png": (
            "Directional Deviation by Input Azimuth",
            "The plot evaluates whether model disagreement changes with the imposed 2D source-probe direction.",
            "The graph is valid only because the input azimuth is computed from shared source-probe geometry and is therefore consistent across models for each case.",
        ),
        "outlier_count_by_model.png": (
            "Outlier Count by Model",
            "The plot summarizes prototype-level numerical warning counts for each model service.",
            "A higher count indicates lower numerical stability or scale consistency in the current implementation, not necessarily a physically impossible response.",
        ),
        "feature_sensitivity_heatmap.png": (
            "Observed Feature-Output Association Heatmap",
            "The heatmap summarizes how strongly each model output varies with physical material parameters across the available 2D experiments.",
            "Because no controlled perturbation study was run here, this figure shows observed association rather than controlled sensitivity.",
        ),
        "max_displacement_without_fno.png": (
            "Maximum Displacement without FNO",
            "The plot compares non-FNO displacement magnitudes after excluding the scale-unstable FNO baseline for readability.",
            "It is included as a diagnostic aid and should be interpreted together with the explicit FNO diagnostic report.",
        ),
        "temperature_perturbation_without_fno.png": (
            "Temperature Perturbation without FNO",
            "The plot compares non-FNO temperature perturbation predictions after excluding the scale-unstable FNO baseline.",
            "This improves readability but does not remove the need to discuss FNO separately as an unstable prototype baseline.",
        ),
        "fno_scale_outlier_diagnostic.png": (
            "FNO Scale Outlier Diagnostic",
            "The plot visualizes the difference in displacement scale between FNO and the other model services.",
            "This diagnostic supports the interpretation that FNO remains numerically unstable in the current prototype and should not be overinterpreted physically.",
        ),
    }
    lines = ["# Figure Captions and Interpretation", ""]
    for path in unique_pngs:
        name = Path(path).name
        title, caption, warning = captions.get(
            name,
            (
                name.replace("_", " ").replace(".png", "").title(),
                "The plot compares model behavior under identical 2D source-probe conditions.",
                "The trend should be interpreted cautiously because the current results come from a research prototype rather than a fully validated field-scale simulation.",
            ),
        )
        lines += [
            f"## {name}",
            "",
            f"- Figure file: `{path}`",
            "- LaTeX:",
            "```latex",
            "\\begin{figure}[ht]",
            "  \\centering",
            f"  \\includegraphics[width=0.9\\textwidth]{{figures/model_comparison/{name}}}",
            f"  \\caption{{{caption}}}",
            f"  \\label{{fig:{Path(name).stem}}}",
            "\\end{figure}",
            "```",
            f"- Interpretation: {title}.",
            f"- Warnings / limitations: {warning}",
            "",
        ]
    if not speed_available:
        lines.append("- Speed-specific figures were skipped because no latency column was available in the current summary dataset.")
    if not reference_available:
        lines.append("- Accuracy-specific figures were skipped because no explicit ground-truth columns were available; agreement and stability analysis were used instead.")
    return "\n".join(lines) + "\n"


def build_model_comparison_final_report(
    summary_df: pd.DataFrame,
    validated_df: pd.DataFrame,
    created_graphs: list[str],
    skipped_graphs: list[str],
    material_source_path: Path,
    summary_path: Path,
    reference_available: bool,
    speed_available: bool,
    fno_warnings: list[str],
) -> str:
    models = sorted(validated_df["model"].dropna().astype(str).unique().tolist(), key=model_sort_key)
    materials = sorted(validated_df["material"].dropna().astype(str).unique().tolist(), key=material_sort_key)
    png_graphs = [path for path in created_graphs if path.endswith(".png")]
    lines = [
        "# Model Comparison Final Report",
        "",
        "## Inputs inspected",
        f"- Summary file: `{summary_path}`",
        f"- Material source: `{material_source_path}`",
        "",
        "## Dataset usage",
        f"- Rows loaded: `{len(summary_df)}`",
        f"- Rows used after 2D validation: `{len(validated_df)}`",
        f"- Materials included: `{', '.join(materials)}`",
        f"- Models included: `{', '.join(models)}`",
        "",
        "## Availability",
        f"- Valid 2D dataset: `{'yes' if len(validated_df) == len(summary_df) else 'partial'}`",
        f"- Reference / ground-truth available: `{'yes' if reference_available else 'no'}`",
        f"- Speed / latency available: `{'yes' if speed_available else 'no'}`",
        "",
        "## Generated graphs",
        f"- PNG graphs created: `{len(png_graphs)}`",
    ]
    lines.extend(f"- `{Path(path).name}`" for path in png_graphs)
    lines += [
        "",
        "## Skipped graph items",
        f"- Count: `{len(skipped_graphs)}`",
    ]
    lines.extend(f"- {item}" for item in skipped_graphs)
    lines += [
        "",
        "## FNO warnings",
    ]
    if fno_warnings:
        lines.extend(f"- {warning}" for warning in fno_warnings)
    else:
        lines.append("- No explicit FNO outlier warnings were triggered.")
    lines += [
        "",
        "## Chapter 5 usage note",
        "Use these figures as comparative prototype results for identical 2D source-probe scenarios.",
        "They support discussion of model agreement, stability, directional behavior, and observed feature-output associations.",
        "They should not be described as fully validated field-scale thermoelastic simulations.",
    ]
    return "\n".join(lines) + "\n"


def write_csv_and_tex(df: pd.DataFrame, csv_path: Path, tex_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    tex_path.write_text(df.to_latex(index=False, escape=True), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def model_sort_key(value: Any) -> int:
    text = normalize_model(value)
    return MODEL_ORDER.index(text) if text in MODEL_ORDER else len(MODEL_ORDER)


def material_sort_key(value: Any) -> int:
    text = normalize_material(value)
    return MATERIAL_ORDER.index(text) if text in MATERIAL_ORDER else len(MATERIAL_ORDER)


def find_reference_like_files() -> list[Path]:
    matches: list[Path] = []
    keywords = ["reference", "target", "ground_truth", "comsol"]
    for root in DEFAULT_REPORTS_SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            name = path.name.lower()
            if any(keyword in name for keyword in keywords):
                matches.append(path)
    unique = sorted(set(matches))
    return unique[:50]


if __name__ == "__main__":
    main()
