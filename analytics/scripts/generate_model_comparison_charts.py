from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


MODEL_COLORS = {
    "meshgraphnet": "#60a5fa",
    "fno": "#34d399",
    "pinn": "#fbbf24",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate model comparison charts from prediction metrics CSV.")
    parser.add_argument("--metrics", default="analytics/outputs/model_comparison_metrics.csv")
    parser.add_argument("--output-dir", default="analytics/charts")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics_path = Path(args.metrics).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(metrics_path)
    if not records:
        raise ValueError(f"No successful prediction rows found in {metrics_path}")

    chart_paths = [
        write_metric_bar_chart(output_dir / "azimuth_comparison.svg", records, "azimuth_deg", "Average Azimuth By Rock And Model"),
        write_metric_bar_chart(output_dir / "travel_time_comparison.svg", records, "travel_time_ms", "Average Travel Time By Rock And Model"),
        write_metric_bar_chart(output_dir / "magnitude_comparison.svg", records, "magnitude", "Average Direction Magnitude By Rock And Model"),
        write_sensitivity_chart(output_dir / "temperature_sensitivity.svg", records, "temperature", "temperature_c", "Temperature Sensitivity"),
        write_sensitivity_chart(output_dir / "pressure_sensitivity.svg", records, "pressure", "pressure_mpa", "Pressure Sensitivity"),
        write_sensitivity_chart(output_dir / "frequency_sensitivity.svg", records, "frequency", "frequency_hz", "Frequency Sensitivity"),
        write_direction_components_chart(output_dir / "direction_components.svg", records),
        write_heatmap(output_dir / "temperature_pressure_heatmap.svg", records),
    ]
    surface_path = write_surface_html(output_dir / "travel_time_3d_surface.html", records)
    dashboard_path = write_dashboard(output_dir / "model_comparison_dashboard.html", chart_paths, surface_path)

    print("Charts directory:", output_dir)
    print("Dashboard:", dashboard_path)


def load_records(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row.get("status") == "ok"]


def write_metric_bar_chart(path: Path, records: list[dict[str, str]], metric: str, title: str) -> Path:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in records:
        value = to_float(row.get(metric))
        if value is not None:
            grouped[(row["medium_id"], row["model"])].append(value)

    media = sorted({medium for medium, _ in grouped})
    models = sorted({model for _, model in grouped})
    labels = media
    series = {
        model: [average(grouped.get((medium, model), [])) for medium in media]
        for model in models
    }
    path.write_text(render_grouped_bar_svg(title, labels, series, ylabel=metric), encoding="utf-8")
    return path


def write_sensitivity_chart(path: Path, records: list[dict[str, str]], sweep_name: str, x_field: str, title: str) -> Path:
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in records:
        if row["sweep_name"] != sweep_name:
            continue
        x_value = to_float(row.get(x_field))
        y_value = to_float(row.get("travel_time_ms"))
        if x_value is not None and y_value is not None:
            grouped[(row["model"], x_value)].append(y_value)

    models = sorted({model for model, _ in grouped})
    x_values = sorted({x_value for _, x_value in grouped})
    series = {
        model: [(x_value, average(grouped.get((model, x_value), []))) for x_value in x_values]
        for model in models
    }
    path.write_text(render_line_svg(title, series, xlabel=x_field, ylabel="travel_time_ms"), encoding="utf-8")
    return path


def write_direction_components_chart(path: Path, records: list[dict[str, str]]) -> Path:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in records:
        for component in ("direction_x", "direction_y", "direction_z"):
            value = to_float(row.get(component))
            if value is not None:
                grouped[(row["model"], component)].append(value)

    components = ["direction_x", "direction_y", "direction_z"]
    models = sorted({model for model, _ in grouped})
    series = {
        model: [average(grouped.get((model, component), [])) for component in components]
        for model in models
    }
    path.write_text(render_grouped_bar_svg("Average Direction Vector Components", components, series, ylabel="component"), encoding="utf-8")
    return path


def write_heatmap(path: Path, records: list[dict[str, str]]) -> Path:
    grouped: dict[tuple[float, float], list[float]] = defaultdict(list)
    for row in records:
        if row["sweep_name"] != "temperature_pressure":
            continue
        temperature = to_float(row.get("temperature_c"))
        pressure = to_float(row.get("pressure_mpa"))
        travel_time = to_float(row.get("travel_time_ms"))
        if temperature is not None and pressure is not None and travel_time is not None:
            grouped[(temperature, pressure)].append(travel_time)

    temperatures = sorted({temperature for temperature, _ in grouped})
    pressures = sorted({pressure for _, pressure in grouped})
    values = {
        key: average(items)
        for key, items in grouped.items()
    }
    path.write_text(render_heatmap_svg("Temperature-Pressure Travel Time Heatmap", temperatures, pressures, values), encoding="utf-8")
    return path


def write_surface_html(path: Path, records: list[dict[str, str]]) -> Path:
    points = []
    for row in records:
        if row["sweep_name"] != "temperature_pressure":
            continue
        temperature = to_float(row.get("temperature_c"))
        pressure = to_float(row.get("pressure_mpa"))
        travel_time = to_float(row.get("travel_time_ms"))
        if temperature is not None and pressure is not None and travel_time is not None:
            points.append((temperature, pressure, travel_time, row["model"], row["medium_id"]))

    rows = "\n".join(
        f"<tr><td>{temp:.4g}</td><td>{pressure:.4g}</td><td>{travel:.6g}</td><td>{html.escape(model)}</td><td>{html.escape(medium)}</td></tr>"
        for temp, pressure, travel, model, medium in points
    )
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Travel Time 3D Surface Data</title>
  <style>
    body {{ background: #0f172a; color: #e5eefb; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; }}
    main {{ width: min(1000px, calc(100% - 48px)); margin: 40px auto; }}
    table {{ width: 100%; border-collapse: collapse; background: #172033; border: 1px solid #334155; border-radius: 16px; overflow: hidden; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #263244; text-align: right; color: #cbd5e1; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: #bfdbfe; }}
  </style>
</head>
<body>
<main>
  <h1>Travel Time Surface Data</h1>
  <p>This table is the source data for a temperature-pressure-travel-time 3D surface.</p>
  <table>
    <thead><tr><th>temperature_c</th><th>pressure_mpa</th><th>travel_time_ms</th><th>model</th><th>medium</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return path


def write_dashboard(path: Path, chart_paths: list[Path], surface_path: Path) -> Path:
    cards = "\n".join(
        f'<section class="card"><h2>{html.escape(chart.stem.replace("_", " ").title())}</h2><img src="{html.escape(chart.name)}" alt="{html.escape(chart.stem)}"></section>'
        for chart in chart_paths
    )
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model Comparison Dashboard</title>
  <style>
    body {{ margin: 0; background: #0f172a; color: #e5eefb; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1180px, calc(100% - 48px)); margin: 40px auto 72px; }}
    h1 {{ letter-spacing: -0.03em; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 18px; }}
    .card {{ background: #172033; border: 1px solid #334155; border-radius: 18px; padding: 18px; }}
    img {{ width: 100%; display: block; }}
    a {{ color: #93c5fd; }}
  </style>
</head>
<body>
<main>
  <h1>Thermoelastic Model Comparison Dashboard</h1>
  <p><a href="{html.escape(surface_path.name)}">Open travel-time surface data</a></p>
  <div class="grid">{cards}</div>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return path


def render_grouped_bar_svg(title: str, labels: list[str], series: dict[str, list[float]], ylabel: str) -> str:
    width, height = 980, 520
    margin_left, margin_right, margin_top, margin_bottom = 78, 30, 64, 88
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    all_values = [value for values in series.values() for value in values if value is not None]
    y_min, y_max = nice_range(all_values)
    model_names = list(series)
    group_w = plot_w / max(len(labels), 1)
    bar_w = group_w / max(len(model_names) + 1, 1)
    parts = svg_header(width, height, title)
    parts.append(axis_svg(margin_left, margin_top, plot_w, plot_h, y_min, y_max, ylabel))
    for label_index, label in enumerate(labels):
        x0 = margin_left + label_index * group_w
        parts.append(svg_text(x0 + group_w / 2, height - 36, label, anchor="middle", size=12, fill="#cbd5e1"))
        for model_index, model in enumerate(model_names):
            value = series[model][label_index]
            if value is None:
                continue
            x = x0 + (model_index + 0.5) * bar_w
            y = scale(value, y_min, y_max, margin_top + plot_h, margin_top)
            h = margin_top + plot_h - y
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w * 0.78:.2f}" height="{h:.2f}" rx="4" fill="{MODEL_COLORS.get(model, "#94a3b8")}"/>')
    parts.extend(legend_svg(model_names, width - margin_right - 180, 22))
    return "\n".join(parts + ["</svg>"])


def render_line_svg(title: str, series: dict[str, list[tuple[float, float | None]]], xlabel: str, ylabel: str) -> str:
    width, height = 980, 520
    margin_left, margin_right, margin_top, margin_bottom = 78, 42, 64, 80
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    all_x = [x for points in series.values() for x, y in points if y is not None]
    all_y = [y for points in series.values() for x, y in points if y is not None]
    x_min, x_max = nice_range(all_x)
    y_min, y_max = nice_range(all_y)
    parts = svg_header(width, height, title)
    parts.append(axis_svg(margin_left, margin_top, plot_w, plot_h, y_min, y_max, ylabel))
    parts.append(svg_text(margin_left + plot_w / 2, height - 24, xlabel, anchor="middle", size=13, fill="#94a3b8"))
    for model, points in series.items():
        coords = [
            (
                scale(x, x_min, x_max, margin_left, margin_left + plot_w),
                scale(y, y_min, y_max, margin_top + plot_h, margin_top),
            )
            for x, y in points
            if y is not None
        ]
        if not coords:
            continue
        path = " ".join(f"{'M' if index == 0 else 'L'} {x:.2f} {y:.2f}" for index, (x, y) in enumerate(coords))
        parts.append(f'<path d="{path}" fill="none" stroke="{MODEL_COLORS.get(model, "#94a3b8")}" stroke-width="3"/>')
        for x, y in coords:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{MODEL_COLORS.get(model, "#94a3b8")}"/>')
    parts.extend(legend_svg(list(series), width - margin_right - 180, 22))
    return "\n".join(parts + ["</svg>"])


def render_heatmap_svg(title: str, temperatures: list[float], pressures: list[float], values: dict[tuple[float, float], float]) -> str:
    width, height = 920, 560
    left, top = 120, 74
    cell_w = 680 / max(len(temperatures), 1)
    cell_h = 360 / max(len(pressures), 1)
    finite_values = list(values.values())
    v_min, v_max = nice_range(finite_values)
    parts = svg_header(width, height, title)
    for p_index, pressure in enumerate(pressures):
        for t_index, temperature in enumerate(temperatures):
            value = values.get((temperature, pressure))
            fill = heat_color(value, v_min, v_max) if value is not None else "#1f2937"
            x = left + t_index * cell_w
            y = top + (len(pressures) - 1 - p_index) * cell_h
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w:.2f}" height="{cell_h:.2f}" fill="{fill}" stroke="#0f172a"/>')
            if value is not None:
                parts.append(svg_text(x + cell_w / 2, y + cell_h / 2 + 4, f"{value:.3g}", anchor="middle", size=11, fill="#f8fafc"))
    for t_index, temperature in enumerate(temperatures):
        parts.append(svg_text(left + t_index * cell_w + cell_w / 2, top + len(pressures) * cell_h + 28, f"{temperature:g}", anchor="middle", size=12, fill="#cbd5e1"))
    for p_index, pressure in enumerate(pressures):
        y = top + (len(pressures) - 1 - p_index) * cell_h + cell_h / 2 + 4
        parts.append(svg_text(left - 16, y, f"{pressure:g}", anchor="end", size=12, fill="#cbd5e1"))
    parts.append(svg_text(left + 340, height - 48, "temperature_c", anchor="middle", size=13, fill="#94a3b8"))
    parts.append(svg_text(36, top + 180, "pressure_mpa", anchor="middle", size=13, fill="#94a3b8", rotate=-90))
    return "\n".join(parts + ["</svg>"])


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" rx="18" fill="#172033"/>',
        svg_text(28, 38, title, size=22, fill="#f8fafc", weight=700),
    ]


def axis_svg(x: int, y: int, width: int, height: int, y_min: float, y_max: float, ylabel: str) -> str:
    lines = [f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="#111827" stroke="#334155" rx="10"/>']
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4
        yy = scale(value, y_min, y_max, y + height, y)
        lines.append(f'<line x1="{x}" x2="{x + width}" y1="{yy:.2f}" y2="{yy:.2f}" stroke="#263244"/>')
        lines.append(svg_text(x - 12, yy + 4, f"{value:.3g}", anchor="end", size=11, fill="#94a3b8"))
    lines.append(svg_text(24, y + height / 2, ylabel, anchor="middle", size=13, fill="#94a3b8", rotate=-90))
    return "\n".join(lines)


def legend_svg(models: list[str], x: float, y: float) -> list[str]:
    parts = []
    for index, model in enumerate(models):
        yy = y + index * 22
        parts.append(f'<circle cx="{x:.2f}" cy="{yy:.2f}" r="6" fill="{MODEL_COLORS.get(model, "#94a3b8")}"/>')
        parts.append(svg_text(x + 14, yy + 4, model, size=12, fill="#cbd5e1"))
    return parts


def svg_text(x: float, y: float, text: str, *, anchor: str = "start", size: int = 12, fill: str = "#cbd5e1", weight: int = 400, rotate: int | None = None) -> str:
    transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ""
    return f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{fill}" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"{transform}>{html.escape(text)}</text>'


def heat_color(value: float, v_min: float, v_max: float) -> str:
    ratio = 0.0 if v_max == v_min else (value - v_min) / (v_max - v_min)
    ratio = max(0.0, min(1.0, ratio))
    r = int(59 + ratio * (251 - 59))
    g = int(130 + ratio * (191 - 130))
    b = int(246 + ratio * (36 - 246))
    return f"#{r:02x}{g:02x}{b:02x}"


def nice_range(values: Iterable[float | None]) -> tuple[float, float]:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return 0.0, 1.0
    minimum = min(finite)
    maximum = max(finite)
    if minimum == maximum:
        padding = abs(minimum) * 0.1 or 1.0
        return minimum - padding, maximum + padding
    padding = (maximum - minimum) * 0.08
    return minimum - padding, maximum + padding


def scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return (dst_min + dst_max) / 2
    return dst_min + (value - src_min) / (src_max - src_min) * (dst_max - dst_min)


def average(values: Iterable[float]) -> float | None:
    items = [value for value in values if value is not None and math.isfinite(value)]
    if not items:
        return None
    return sum(items) / len(items)


def to_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


if __name__ == "__main__":
    main()
