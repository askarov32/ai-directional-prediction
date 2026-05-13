from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


SVG_WIDTH = 980
SVG_HEIGHT = 320
PLOT_PADDING = 44


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an HTML/SVG report from PINN training metrics.")
    parser.add_argument(
        "--metrics-json",
        default="pinn-service/artifacts/checkpoints/baseline/metrics.json",
        help="Path to metrics.json produced by training.",
    )
    parser.add_argument(
        "--metrics-csv",
        default=None,
        help="Optional path to metrics.csv. If omitted, the script tries metrics.csv next to metrics.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for report artifacts. Defaults to <checkpoint-dir>/report.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics_json_path = Path(args.metrics_json).expanduser().resolve()
    metrics_csv_path = resolve_metrics_csv_path(metrics_json_path, args.metrics_csv)
    output_dir = resolve_output_dir(metrics_json_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_payload = json.loads(metrics_json_path.read_text(encoding="utf-8"))
    history = load_history(metrics_json_path=metrics_json_path, metrics_csv_path=metrics_csv_path, json_payload=json_payload)
    summary = build_summary(history=history, json_payload=json_payload, metrics_json_path=metrics_json_path, metrics_csv_path=metrics_csv_path)

    charts = {
        "total_loss": build_svg_chart(
            title="Total Loss",
            series=[
                ("train_total_loss", extract_series(history, "total_loss"), "#60a5fa"),
                ("val_total_loss", extract_series(history, "val_total_loss"), "#f59e0b"),
            ],
        ),
        "component_loss": build_svg_chart(
            title="Core Loss Components",
            series=[
                ("supervised_loss", extract_series(history, "supervised_loss"), "#60a5fa"),
                ("wave_residual_loss", extract_series(history, "wave_residual_loss"), "#34d399"),
                ("thermal_residual_loss", extract_series(history, "thermal_residual_loss"), "#f59e0b"),
            ],
        ),
        "normalized_loss": build_svg_chart(
            title="Normalized Loss Components",
            series=[
                ("normalized_supervised_loss", extract_series(history, "normalized_supervised_loss"), "#60a5fa"),
                ("normalized_wave_residual_loss", extract_series(history, "normalized_wave_residual_loss"), "#34d399"),
                ("normalized_thermal_residual_loss", extract_series(history, "normalized_thermal_residual_loss"), "#f59e0b"),
            ],
        ),
        "optimization": build_svg_chart(
            title="Learning Rate And Gradient Norm",
            series=[
                ("learning_rate", extract_series(history, "learning_rate"), "#a78bfa"),
                ("grad_norm", extract_series(history, "grad_norm"), "#f472b6"),
            ],
        ),
    }

    chart_paths = {}
    for key, svg in charts.items():
        path = output_dir / f"{key}.svg"
        path.write_text(svg, encoding="utf-8")
        chart_paths[key] = path

    summary_path = output_dir / "training_report_summary.json"
    html_path = output_dir / "training_report.html"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    html_path.write_text(render_html(summary=summary, chart_paths=chart_paths), encoding="utf-8")

    print("Training report summary:", summary_path)
    print("Training report HTML:", html_path)
    for key, path in chart_paths.items():
        print(f"{key}_chart:", path)


def resolve_metrics_csv_path(metrics_json_path: Path, raw_value: str | None) -> Path | None:
    if raw_value:
        candidate = Path(raw_value).expanduser().resolve()
        return candidate if candidate.exists() else None
    candidate = metrics_json_path.with_name("metrics.csv")
    return candidate if candidate.exists() else None


def resolve_output_dir(metrics_json_path: Path, raw_value: str | None) -> Path:
    if raw_value:
        return Path(raw_value).expanduser().resolve()
    return metrics_json_path.parent / "report"


def load_history(*, metrics_json_path: Path, metrics_csv_path: Path | None, json_payload: dict[str, Any]) -> list[dict[str, float]]:
    if metrics_csv_path and metrics_csv_path.exists():
        rows = []
        with metrics_csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                parsed = {}
                for key, value in row.items():
                    parsed[key] = to_float(value)
                rows.append(parsed)
        if rows:
            return rows

    history = json_payload.get("history", [])
    if not isinstance(history, list):
        raise ValueError(f"metrics history must be a list in {metrics_json_path}")
    normalized = []
    for item in history:
        if not isinstance(item, dict):
            continue
        normalized.append({key: to_float(value) for key, value in item.items()})
    return normalized


def build_summary(
    *,
    history: list[dict[str, float]],
    json_payload: dict[str, Any],
    metrics_json_path: Path,
    metrics_csv_path: Path | None,
) -> dict[str, Any]:
    last = history[-1] if history else {}
    first = history[0] if history else {}
    validation_present = any("val_total_loss" in row and row["val_total_loss"] is not None for row in history)
    learning_rate_changed = any(row.get("learning_rate") not in (None, first.get("learning_rate")) for row in history[1:])

    return {
        "metrics_json_path": str(metrics_json_path),
        "metrics_csv_path": str(metrics_csv_path) if metrics_csv_path else None,
        "epochs_recorded": len(history),
        "validation_enabled": bool(json_payload.get("validation_enabled", validation_present)),
        "stopped_early": bool(json_payload.get("stopped_early", False)),
        "completed_epochs": int(json_payload.get("completed_epochs", len(history))),
        "best_metric_name": json_payload.get("best_metric", "total_loss"),
        "best_loss": to_float(json_payload.get("best_loss")),
        "initial_total_loss": first.get("total_loss"),
        "final_total_loss": last.get("total_loss"),
        "initial_val_total_loss": first.get("val_total_loss"),
        "final_val_total_loss": last.get("val_total_loss"),
        "initial_learning_rate": first.get("learning_rate"),
        "final_learning_rate": last.get("learning_rate"),
        "max_grad_norm": max_value(history, "grad_norm"),
        "min_best_so_far": min_value(history, "best_so_far"),
        "learning_rate_changed": learning_rate_changed,
        "epochs_without_improvement": int(last.get("epochs_without_improvement") or 0),
        "latest_epoch": int(last.get("epoch") or 0),
    }


def extract_series(history: list[dict[str, float]], field: str) -> list[tuple[float, float]]:
    series = []
    for row in history:
        epoch = row.get("epoch")
        value = row.get(field)
        if epoch is None or value is None:
            continue
        series.append((epoch, value))
    return series


def build_svg_chart(*, title: str, series: list[tuple[str, list[tuple[float, float]], str]]) -> str:
    active_series = [(name, values, color) for name, values, color in series if values]
    if not active_series:
        return render_empty_chart(title)

    all_points = [point for _, values, _ in active_series for point in values]
    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points]

    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)
    if min_x == max_x:
        max_x = min_x + 1.0
    if min_y == max_y:
        max_y = min_y + 1.0

    plot_width = SVG_WIDTH - (PLOT_PADDING * 2)
    plot_height = SVG_HEIGHT - (PLOT_PADDING * 2)

    def project_x(value: float) -> float:
        return PLOT_PADDING + ((value - min_x) / (max_x - min_x)) * plot_width

    def project_y(value: float) -> float:
        return SVG_HEIGHT - PLOT_PADDING - ((value - min_y) / (max_y - min_y)) * plot_height

    line_paths = []
    legend_items = []
    for index, (name, values, color) in enumerate(active_series):
        coordinates = " ".join(f"{project_x(x):.2f},{project_y(y):.2f}" for x, y in values)
        line_paths.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{coordinates}" />'
        )
        legend_y = 24 + (index * 18)
        legend_items.append(
            f'<g><rect x="{SVG_WIDTH - 220}" y="{legend_y - 9}" width="12" height="12" fill="{color}" rx="2" />'
            f'<text x="{SVG_WIDTH - 200}" y="{legend_y}" fill="#cbd5e1" font-size="12">{html.escape(name)}</text></g>'
        )

    y_ticks = []
    for step in range(5):
        fraction = step / 4 if 4 else 0
        value = max_y - ((max_y - min_y) * fraction)
        y = PLOT_PADDING + (plot_height * fraction)
        y_ticks.append(
            f'<g><line x1="{PLOT_PADDING}" y1="{y:.2f}" x2="{SVG_WIDTH - PLOT_PADDING}" y2="{y:.2f}" stroke="#243042" stroke-width="1" />'
            f'<text x="8" y="{y + 4:.2f}" fill="#94a3b8" font-size="11">{format_number(value)}</text></g>'
        )

    x_ticks = []
    for step in range(5):
        fraction = step / 4 if 4 else 0
        value = min_x + ((max_x - min_x) * fraction)
        x = PLOT_PADDING + (plot_width * fraction)
        x_ticks.append(
            f'<g><line x1="{x:.2f}" y1="{SVG_HEIGHT - PLOT_PADDING}" x2="{x:.2f}" y2="{PLOT_PADDING}" stroke="#1b2534" stroke-width="1" />'
            f'<text x="{x - 10:.2f}" y="{SVG_HEIGHT - 12}" fill="#94a3b8" font-size="11">{format_number(value)}</text></g>'
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">
  <rect width="{SVG_WIDTH}" height="{SVG_HEIGHT}" rx="18" fill="#111827" />
  <text x="{PLOT_PADDING}" y="24" fill="#f8fafc" font-size="18">{html.escape(title)}</text>
  {''.join(y_ticks)}
  {''.join(x_ticks)}
  <rect x="{PLOT_PADDING}" y="{PLOT_PADDING}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#334155" stroke-width="1" />
  {''.join(line_paths)}
  {''.join(legend_items)}
</svg>
"""


def render_empty_chart(title: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">
  <rect width="{SVG_WIDTH}" height="{SVG_HEIGHT}" rx="18" fill="#111827" />
  <text x="{PLOT_PADDING}" y="24" fill="#f8fafc" font-size="18">{html.escape(title)}</text>
  <text x="{PLOT_PADDING}" y="72" fill="#94a3b8" font-size="14">No data available for this chart.</text>
</svg>
"""


def render_html(*, summary: dict[str, Any], chart_paths: dict[str, Path]) -> str:
    metric_cards = [
        ("epochs_recorded", summary.get("epochs_recorded")),
        ("best_metric", summary.get("best_metric_name")),
        ("best_loss", summary.get("best_loss")),
        ("final_total_loss", summary.get("final_total_loss")),
        ("final_val_total_loss", summary.get("final_val_total_loss")),
        ("final_learning_rate", summary.get("final_learning_rate")),
        ("max_grad_norm", summary.get("max_grad_norm")),
        ("stopped_early", summary.get("stopped_early")),
    ]
    cards = "".join(
        f'<div class="metric"><span>{html.escape(label)}</span><strong>{html.escape(format_value(value))}</strong></div>'
        for label, value in metric_cards
    )
    chart_blocks = "".join(
        f'<section class="chart"><img src="{path.name}" alt="{html.escape(name)} chart"></section>'
        for name, path in chart_paths.items()
    )
    notes = []
    if summary.get("validation_enabled"):
        notes.append("Validation metrics are present and best checkpoint selection follows validation loss.")
    else:
        notes.append("This run has no validation split, so checkpoint selection follows training total loss.")
    if summary.get("learning_rate_changed"):
        notes.append("Learning rate changed during training, which indicates the plateau scheduler activated.")
    if summary.get("stopped_early"):
        notes.append("Training stopped early before the requested epoch budget was exhausted.")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PINN Training Report</title>
  <style>
    body {{ margin: 0; background: #0f172a; color: #e5eefb; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1200px, calc(100% - 48px)); margin: 40px auto 72px; }}
    h1, h2, p {{ margin: 0; }}
    h1 {{ font-size: 34px; }}
    h2 {{ font-size: 22px; margin-top: 28px; }}
    p {{ color: #94a3b8; margin-top: 10px; }}
    .panel {{ background: #172033; border: 1px solid #334155; border-radius: 18px; padding: 22px; margin-top: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .metric {{ background: #111827; border: 1px solid #263244; border-radius: 14px; padding: 14px; }}
    .metric span {{ display: block; color: #94a3b8; font-size: 12px; }}
    .metric strong {{ display: block; margin-top: 6px; font-size: 18px; color: #f8fafc; }}
    .chart-grid {{ display: grid; gap: 16px; margin-top: 18px; }}
    .chart img {{ display: block; width: 100%; height: auto; border-radius: 14px; border: 1px solid #263244; background: #111827; }}
    ul {{ margin: 14px 0 0; padding-left: 18px; color: #cbd5e1; }}
    code {{ color: #dbeafe; }}
  </style>
</head>
<body>
<main>
  <h1>PINN Training Report</h1>
  <p>Metrics source: <code>{html.escape(summary["metrics_json_path"])}</code></p>
  <section class="panel">
    <h2>Overview</h2>
    <div class="grid">{cards}</div>
  </section>
  <section class="panel">
    <h2>Reading Notes</h2>
    <ul>{''.join(f'<li>{html.escape(note)}</li>' for note in notes)}</ul>
  </section>
  <section class="panel">
    <h2>Charts</h2>
    <div class="chart-grid">{chart_blocks}</div>
  </section>
</main>
</body>
</html>
"""


def max_value(history: list[dict[str, float]], field: str) -> float | None:
    values = [row[field] for row in history if row.get(field) is not None]
    return max(values) if values else None


def min_value(history: list[dict[str, float]], field: str) -> float | None:
    values = [row[field] for row in history if row.get(field) is not None]
    return min(values) if values else None


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4g}"


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and value is not None:
        return format_number(float(value))
    if value is None:
        return "n/a"
    return str(value)


if __name__ == "__main__":
    main()
