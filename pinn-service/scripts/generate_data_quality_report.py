from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np


FIELD_GROUPS = {
    "material_static": ["youngs_modulus", "poissons_ratio", "density", "thermal_expansion"],
    "thermal_properties": ["thermal_conductivity", "thermal_density", "heat_capacity"],
    "temperature": ["temperature_k"],
    "displacement": ["disp_x", "disp_y", "disp_z"],
    "velocity": ["vel_x", "vel_y", "vel_z"],
    "stress_normal": ["von_mises", "stress_x", "stress_y", "stress_z"],
    "stress_shear": ["stress_xy", "stress_yz", "stress_xz"],
    "strain": ["strain_x", "strain_y", "strain_z", "strain_xy", "strain_yz", "strain_xz"],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate data quality JSON/HTML reports for rod experiment datasets.")
    parser.add_argument(
        "--manifest",
        default="pinn-service/artifacts/rod_experiments/manifest.json",
        help="Path to rod experiment manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="pinn-service/artifacts/rod_experiments/reports",
        help="Directory where data_quality_report.json/html will be written.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = build_report(manifest_path)

    json_path = output_dir / "data_quality_report.json"
    html_path = output_dir / "data_quality_report.html"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")

    print("Data quality JSON:", json_path)
    print("Data quality HTML:", html_path)


def build_report(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiments: list[dict[str, Any]] = []

    for experiment in manifest.get("experiments", []):
        metadata_path = Path(experiment["metadata"]).expanduser().resolve()
        structured_path = Path(experiment["structured_dataset"]).expanduser().resolve()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload = np.load(structured_path)

        groups = {}
        for array_name, component_names in FIELD_GROUPS.items():
            if array_name not in payload:
                continue
            groups[array_name] = summarize_array(payload[array_name], component_names)

        experiments.append(
            {
                "rock_id": experiment["rock_id"],
                "experiment_id": metadata.get("experiment_id"),
                "metadata_path": str(metadata_path),
                "structured_dataset": str(structured_path),
                "node_count": metadata.get("node_count"),
                "raw_node_counts": metadata.get("raw_node_counts"),
                "dropped_node_counts": metadata.get("dropped_node_counts"),
                "duplicate_coordinate_counts": metadata.get("duplicate_coordinate_counts"),
                "coordinate_policy": metadata.get("coordinate_policy"),
                "time_steps": metadata.get("time_steps"),
                "time_start": metadata.get("time_start"),
                "time_end": metadata.get("time_end"),
                "time_step": metadata.get("time_step"),
                "reference_temperature_k": metadata.get("reference_temperature_k"),
                "strain_source": metadata.get("strain_source"),
                "groups": groups,
            }
        )

    return {
        "manifest": str(manifest_path),
        "experiment_count": len(experiments),
        "experiments": experiments,
    }


def summarize_array(values: np.ndarray, component_names: list[str]) -> dict[str, Any]:
    array = np.asarray(values)
    summary: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "total_values": int(array.size),
        "nan_count": int(np.isnan(array).sum()) if np.issubdtype(array.dtype, np.floating) else 0,
        "inf_count": int(np.isinf(array).sum()) if np.issubdtype(array.dtype, np.floating) else 0,
        "components": {},
    }

    if array.ndim == 1:
        component_arrays = [array]
    elif array.ndim == 2 and len(component_names) == 1:
        component_arrays = [array]
    else:
        component_arrays = [array[..., index] for index in range(array.shape[-1])]

    for name, component in zip(component_names, component_arrays, strict=False):
        finite = component[np.isfinite(component)]
        if finite.size == 0:
            summary["components"][name] = {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "nan_count": int(np.isnan(component).sum()),
                "inf_count": int(np.isinf(component).sum()),
            }
            continue
        summary["components"][name] = {
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "nan_count": int(np.isnan(component).sum()),
            "inf_count": int(np.isinf(component).sum()),
        }
    return summary


def render_html(report: dict[str, Any]) -> str:
    sections = []
    for experiment in report["experiments"]:
        sections.append(render_experiment(experiment))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PINN Rod Data Quality Report</title>
  <style>
    body {{ margin: 0; background: #0f172a; color: #e5eefb; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1180px, calc(100% - 48px)); margin: 40px auto 72px; }}
    h1, h2, h3 {{ margin: 0; }}
    h1 {{ font-size: 34px; letter-spacing: -0.03em; }}
    h2 {{ margin-top: 34px; font-size: 24px; }}
    h3 {{ margin-top: 24px; color: #bfdbfe; }}
    p, td, th {{ color: #cbd5e1; }}
    .muted {{ color: #94a3b8; }}
    .card {{ background: #172033; border: 1px solid #334155; border-radius: 18px; padding: 22px; margin-top: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }}
    .metric {{ background: #111827; border: 1px solid #263244; border-radius: 14px; padding: 14px; }}
    .metric span {{ display: block; color: #94a3b8; font-size: 13px; }}
    .metric strong {{ display: block; margin-top: 6px; font-size: 18px; color: #f8fafc; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; overflow: hidden; border-radius: 14px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #263244; text-align: right; font-size: 13px; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: #bfdbfe; font-weight: 650; background: #111827; }}
    code {{ color: #dbeafe; }}
  </style>
</head>
<body>
<main>
  <h1>PINN Rod Data Quality Report</h1>
  <p class="muted">Experiments: {report["experiment_count"]}. Source manifest: <code>{html.escape(report["manifest"])}</code></p>
  {''.join(sections)}
</main>
</body>
</html>
"""


def render_experiment(experiment: dict[str, Any]) -> str:
    cards = [
        ("Nodes", experiment.get("node_count")),
        ("Time steps", experiment.get("time_steps")),
        ("Time range", f'{experiment.get("time_start")} to {experiment.get("time_end")}'),
        ("Reference T, K", experiment.get("reference_temperature_k")),
        ("Coordinate policy", experiment.get("coordinate_policy")),
        ("Strain source", experiment.get("strain_source")),
    ]
    group_tables = []
    for group_name, group in experiment["groups"].items():
        group_tables.append(f"<h3>{html.escape(group_name)}</h3>{render_group_table(group)}")

    return f"""
  <section class="card">
    <h2>{html.escape(str(experiment["rock_id"]))}</h2>
    <p class="muted">Experiment: <code>{html.escape(str(experiment.get("experiment_id")))}</code></p>
    <div class="grid">
      {''.join(f'<div class="metric"><span>{html.escape(label)}</span><strong>{html.escape(str(value))}</strong></div>' for label, value in cards)}
    </div>
    <h3>Alignment</h3>
    {render_counts_table(experiment)}
    {''.join(group_tables)}
  </section>
"""


def render_counts_table(experiment: dict[str, Any]) -> str:
    raw = experiment.get("raw_node_counts") or {}
    dropped = experiment.get("dropped_node_counts") or {}
    duplicates = experiment.get("duplicate_coordinate_counts") or {}
    rows = []
    for name in raw:
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{raw.get(name)}</td>"
            f"<td>{dropped.get(name)}</td>"
            f"<td>{duplicates.get(name)}</td>"
            "</tr>"
        )
    return f"""<table>
      <thead><tr><th>file group</th><th>raw nodes</th><th>dropped</th><th>duplicates</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


def render_group_table(group: dict[str, Any]) -> str:
    rows = []
    for name, stats in group["components"].items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{format_number(stats['min'])}</td>"
            f"<td>{format_number(stats['max'])}</td>"
            f"<td>{format_number(stats['mean'])}</td>"
            f"<td>{format_number(stats['std'])}</td>"
            f"<td>{stats['nan_count']}</td>"
            f"<td>{stats['inf_count']}</td>"
            "</tr>"
        )
    return f"""<table>
      <thead><tr><th>component</th><th>min</th><th>max</th><th>mean</th><th>std</th><th>NaN</th><th>Inf</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


def format_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}"


if __name__ == "__main__":
    main()
