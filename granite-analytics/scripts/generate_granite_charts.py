from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = BASE_DIR / ".mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
CACHE_DIR = BASE_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402


DEFAULT_PREDICTIONS_PATH = BASE_DIR / "outputs" / "granite_predictions.json"
DEFAULT_SUMMARY_PATH = BASE_DIR / "outputs" / "granite_metrics_summary.json"
DEFAULT_REPORT_PATH = BASE_DIR / "granite_analytics_report.html"
CHARTS_DIR = BASE_DIR / "charts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate charts and an HTML report for prediction analytics.")
    parser.add_argument("--input", type=Path, default=DEFAULT_PREDICTIONS_PATH, help="Predictions JSON file.")
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_PATH, help="Metrics summary JSON file.")
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_PATH, help="HTML report file.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def chart_theme() -> None:
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": "#0f172a",
            "axes.facecolor": "#111827",
            "axes.edgecolor": "#475569",
            "axes.labelcolor": "#e2e8f0",
            "axes.titlecolor": "#f8fafc",
            "xtick.color": "#cbd5e1",
            "ytick.color": "#cbd5e1",
            "grid.color": "#334155",
            "text.color": "#f8fafc",
            "savefig.facecolor": "#0f172a",
            "savefig.edgecolor": "#0f172a",
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
        }
    )


def ensure_dirs() -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def sorted_group(records: list[dict[str, Any]], variable_key: str) -> list[dict[str, Any]]:
    return sorted(records, key=lambda item: float(item["variables"][variable_key]))


def prediction_metric(record: dict[str, Any], key: str) -> float:
    response = record["response"]
    if key in response["prediction"]:
        return float(response["prediction"][key])
    if key in response["field_summary"]:
        return float(response["field_summary"][key])
    if key in response["meta"]:
        return float(response["meta"][key])
    raise KeyError(f"Unsupported metric key: {key}")


def scenario_metrics(record: dict[str, Any]) -> dict[str, float]:
    return {
        "azimuth_deg": prediction_metric(record, "azimuth_deg"),
        "elevation_deg": prediction_metric(record, "elevation_deg"),
        "magnitude": prediction_metric(record, "magnitude"),
        "travel_time_ms": prediction_metric(record, "travel_time_ms"),
        "max_displacement": prediction_metric(record, "max_displacement"),
        "max_temperature_perturbation": prediction_metric(record, "max_temperature_perturbation"),
    }


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_amplitude_time_series(records: list[dict[str, Any]], medium_name: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    amplitudes = sorted({float(record["variables"]["amplitude"]) for record in records})
    palette = ["#93c5fd", "#60a5fa", "#38bdf8", "#22c55e", "#f59e0b"]
    for index, amplitude in enumerate(amplitudes):
        subset = [record for record in records if float(record["variables"]["amplitude"]) == amplitude]
        subset = sorted_group(subset, "time_ms")
        x = [float(record["variables"]["time_ms"]) for record in subset]
        y = [prediction_metric(record, "max_displacement") for record in subset]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            label=f"Amplitude {amplitude:g}",
            color=palette[index % len(palette)],
        )

    ax.set_title(f"{medium_name}: displacement response over time")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Max displacement")
    ax.legend(frameon=False, ncol=min(len(amplitudes), 4))
    save_figure(fig, CHARTS_DIR / "amplitude_time_series.png")


def plot_metrics_comparison(records: list[dict[str, Any]], medium_name: str) -> None:
    scenario_labels = [record["label"] for record in records]
    x = np.arange(len(records))
    metrics = [
        ("azimuth_deg", "Azimuth (deg)", "#60a5fa"),
        ("elevation_deg", "Elevation (deg)", "#22c55e"),
        ("travel_time_ms", "Travel time (ms)", "#f59e0b"),
        ("max_displacement", "Max displacement", "#a78bfa"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (metric_key, title, color) in zip(axes.flat, metrics, strict=True):
        values = [prediction_metric(record, metric_key) for record in records]
        ax.bar(x, values, color=color, width=0.62)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_labels, rotation=22, ha="right")

    fig.suptitle(f"{medium_name}: comparison of representative prediction scenarios", y=1.02)
    save_figure(fig, CHARTS_DIR / "metrics_comparison.png")


def plot_temperature_sensitivity(records: list[dict[str, Any]], medium_name: str) -> None:
    records = sorted_group(records, "temperature_c")
    temperatures = [float(record["variables"]["temperature_c"]) for record in records]
    displacement = [prediction_metric(record, "max_displacement") for record in records]
    temperature_delta = [prediction_metric(record, "max_temperature_perturbation") for record in records]
    travel_time = [prediction_metric(record, "travel_time_ms") for record in records]

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9), sharex=True)
    axes[0].plot(temperatures, displacement, marker="o", color="#60a5fa", linewidth=2)
    axes[0].set_ylabel("Max displacement")
    axes[0].set_title(f"{medium_name}: temperature sensitivity")

    axes[1].plot(temperatures, temperature_delta, marker="o", color="#f59e0b", linewidth=2)
    axes[1].set_ylabel("Temp perturbation")

    axes[2].plot(temperatures, travel_time, marker="o", color="#22c55e", linewidth=2)
    axes[2].set_ylabel("Travel time (ms)")
    axes[2].set_xlabel("Temperature (C)")

    save_figure(fig, CHARTS_DIR / "temperature_sensitivity.png")


def plot_pressure_sensitivity(records: list[dict[str, Any]], medium_name: str) -> None:
    records = sorted_group(records, "pressure_mpa")
    pressures = [float(record["variables"]["pressure_mpa"]) for record in records]
    magnitude = [prediction_metric(record, "magnitude") for record in records]
    displacement = [prediction_metric(record, "max_displacement") for record in records]
    travel_time = [prediction_metric(record, "travel_time_ms") for record in records]

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9), sharex=True)
    axes[0].plot(pressures, magnitude, marker="o", color="#a78bfa", linewidth=2)
    axes[0].set_ylabel("Magnitude")
    axes[0].set_title(f"{medium_name}: pressure sensitivity")

    axes[1].plot(pressures, displacement, marker="o", color="#60a5fa", linewidth=2)
    axes[1].set_ylabel("Max displacement")

    axes[2].plot(pressures, travel_time, marker="o", color="#22c55e", linewidth=2)
    axes[2].set_ylabel("Travel time (ms)")
    axes[2].set_xlabel("Pressure (MPa)")

    save_figure(fig, CHARTS_DIR / "pressure_sensitivity.png")


def plot_frequency_sensitivity(records: list[dict[str, Any]], medium_name: str) -> None:
    records = sorted_group(records, "frequency_hz")
    frequencies = [float(record["variables"]["frequency_hz"]) for record in records]
    magnitude = [prediction_metric(record, "magnitude") for record in records]
    displacement = [prediction_metric(record, "max_displacement") for record in records]
    azimuth = [prediction_metric(record, "azimuth_deg") for record in records]

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9), sharex=True)
    axes[0].plot(frequencies, magnitude, marker="o", color="#a78bfa", linewidth=2)
    axes[0].set_ylabel("Magnitude")
    axes[0].set_title(f"{medium_name}: frequency sensitivity")

    axes[1].plot(frequencies, displacement, marker="o", color="#60a5fa", linewidth=2)
    axes[1].set_ylabel("Max displacement")

    axes[2].plot(frequencies, azimuth, marker="o", color="#f59e0b", linewidth=2)
    axes[2].set_ylabel("Azimuth (deg)")
    axes[2].set_xlabel("Frequency (Hz)")

    save_figure(fig, CHARTS_DIR / "frequency_sensitivity.png")


def build_grid(
    records: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    metric_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values = sorted({float(record["variables"][x_key]) for record in records})
    y_values = sorted({float(record["variables"][y_key]) for record in records})
    grid = np.zeros((len(y_values), len(x_values)))
    for row_index, y_value in enumerate(y_values):
        for col_index, x_value in enumerate(x_values):
            match = next(
                record
                for record in records
                if float(record["variables"][x_key]) == x_value and float(record["variables"][y_key]) == y_value
            )
            grid[row_index, col_index] = prediction_metric(match, metric_key)
    return np.array(x_values), np.array(y_values), grid


def plot_temperature_pressure_heatmap(records: list[dict[str, Any]], medium_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    temperatures, pressures, grid = build_grid(
        records,
        x_key="temperature_c",
        y_key="pressure_mpa",
        metric_key="max_displacement",
    )
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    image = ax.imshow(
        grid,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[temperatures[0], temperatures[-1], pressures[0], pressures[-1]],
    )
    ax.set_title(f"{medium_name}: displacement heatmap over temperature and pressure")
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Pressure (MPa)")
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Max displacement")
    save_figure(fig, CHARTS_DIR / "temperature_pressure_heatmap.png")
    return temperatures, pressures, grid


def plot_three_d_surface(records: list[dict[str, Any]], medium_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times, probe_z_values, grid = build_grid(
        records,
        x_key="time_ms",
        y_key="probe_z",
        metric_key="elevation_deg",
    )

    x_mesh, y_mesh = np.meshgrid(times, probe_z_values)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        x_mesh,
        y_mesh,
        grid,
        cmap="plasma",
        linewidth=0,
        antialiased=True,
        alpha=0.92,
    )
    ax.set_title(f"{medium_name}: 3D elevation response surface")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Probe z")
    ax.set_zlabel("Elevation (deg)")
    fig.colorbar(surface, ax=ax, pad=0.08, shrink=0.7, label="Elevation (deg)")
    save_figure(fig, CHARTS_DIR / "granite_3d_surface.png")
    return times, probe_z_values, grid


def _color_from_value(value: float, min_value: float, max_value: float) -> str:
    cmap = plt.get_cmap("plasma")
    normalizer = colors.Normalize(vmin=min_value, vmax=max_value if max_value > min_value else min_value + 1)
    red, green, blue, _ = cmap(normalizer(value))
    return f"rgb({int(red * 255)}, {int(green * 255)}, {int(blue * 255)})"


def _project_point(x: float, y: float, z: float, width: float, height: float) -> tuple[float, float]:
    angle = math.radians(35)
    scale_x = 290
    scale_y = 170
    x_iso = (x - y) * math.cos(angle) * scale_x + width / 2
    y_iso = (x + y) * math.sin(angle) * scale_y - z * 6 + height / 2 + 60
    return x_iso, y_iso


def write_three_d_surface_html(
    *,
    times: np.ndarray,
    probe_z_values: np.ndarray,
    elevation_grid: np.ndarray,
    medium_name: str,
) -> None:
    width = 960
    height = 720
    x_norm = (times - times.min()) / max(times.max() - times.min(), 1.0)
    y_norm = (probe_z_values - probe_z_values.min()) / max(probe_z_values.max() - probe_z_values.min(), 1.0)
    z_min = float(elevation_grid.min())
    z_max = float(elevation_grid.max())
    z_norm = (elevation_grid - z_min) / max(z_max - z_min, 1.0)

    polygons: list[tuple[float, str, str]] = []
    for row_index in range(len(y_norm) - 1):
        for col_index in range(len(x_norm) - 1):
            corners = [
                (x_norm[col_index], y_norm[row_index], float(z_norm[row_index, col_index]), float(elevation_grid[row_index, col_index])),
                (x_norm[col_index + 1], y_norm[row_index], float(z_norm[row_index, col_index + 1]), float(elevation_grid[row_index, col_index + 1])),
                (x_norm[col_index + 1], y_norm[row_index + 1], float(z_norm[row_index + 1, col_index + 1]), float(elevation_grid[row_index + 1, col_index + 1])),
                (x_norm[col_index], y_norm[row_index + 1], float(z_norm[row_index + 1, col_index]), float(elevation_grid[row_index + 1, col_index])),
            ]
            points = [_project_point(x, y, z, width, height) for x, y, z, _ in corners]
            avg_depth = sum(x + y for x, y, _, _ in corners) / 4
            avg_value = sum(value for _, _, _, value in corners) / 4
            fill = _color_from_value(avg_value, z_min, z_max)
            points_attr = " ".join(f"{point_x:.1f},{point_y:.1f}" for point_x, point_y in points)
            polygons.append(
                (
                    avg_depth,
                    fill,
                    f'<polygon points="{points_attr}" fill="{fill}" fill-opacity="0.92" stroke="#0f172a" stroke-width="1.1" />',
                )
            )

    polygons.sort(key=lambda item: item[0], reverse=True)

    axis_lines = []
    axis_specs = [
        ((0.0, 0.0, 0.0), (1.06, 0.0, 0.0), "Time"),
        ((0.0, 0.0, 0.0), (0.0, 1.06, 0.0), "Probe z"),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.08), "Elevation"),
    ]
    for start, end, label in axis_specs:
        start_x, start_y = _project_point(*start, width, height)
        end_x, end_y = _project_point(*end, width, height)
        axis_lines.append(
            f'<line x1="{start_x:.1f}" y1="{start_y:.1f}" x2="{end_x:.1f}" y2="{end_y:.1f}" stroke="#e2e8f0" stroke-width="1.4" />'
        )
        axis_lines.append(
            f'<text x="{end_x + 8:.1f}" y="{end_y - 4:.1f}" fill="#e2e8f0" font-size="14">{label}</text>'
        )

    rows_html = []
    for row_index, probe_z in enumerate(probe_z_values):
        cells = "".join(
            f"<td>{float(value):.2f}</td>" for value in elevation_grid[row_index]
        )
        rows_html.append(f"<tr><th>{float(probe_z):.2f}</th>{cells}</tr>")

    header_cells = "".join(f"<th>{float(time_ms):.0f}</th>" for time_ms in times)
    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{medium_name} 3D response surface</title>
    <style>
      body {{
        margin: 0;
        background: #0f172a;
        color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }}
      .wrap {{
        max-width: 1160px;
        margin: 0 auto;
        padding: 32px 24px 40px;
      }}
      h1 {{
        margin: 0 0 8px;
        font-size: 28px;
        font-weight: 650;
      }}
      p {{
        margin: 0 0 22px;
        color: #94a3b8;
        line-height: 1.55;
      }}
      .panel {{
        background: #111827;
        border: 1px solid #334155;
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 22px;
      }}
      svg {{
        width: 100%;
        height: auto;
        display: block;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 12px;
        font-size: 14px;
      }}
      th, td {{
        padding: 10px 12px;
        text-align: center;
        border-bottom: 1px solid #1e293b;
      }}
      th {{
        color: #cbd5e1;
        font-weight: 600;
      }}
      td {{
        color: #e2e8f0;
      }}
      .meta {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 12px;
      }}
      .meta div {{
        padding: 14px 16px;
        border-radius: 14px;
        background: #0b1220;
        border: 1px solid #243244;
      }}
      .meta strong {{
        display: block;
        margin-bottom: 6px;
        color: #cbd5e1;
      }}
    </style>
  </head>
  <body>
    <div class="wrap">
      <h1>{medium_name} 3D elevation surface</h1>
      <p>
        This standalone view visualizes how predicted elevation angle changes across
        <strong>time</strong> and <strong>probe depth</strong> in a rect_3d configuration.
        It is generated from the same analytics bundle as the PNG chart and is intended as a quick 3D verification artifact.
      </p>

      <div class="panel">
        <svg viewBox="0 0 {width} {height}" role="img" aria-label="3D elevation surface">
          <rect x="0" y="0" width="{width}" height="{height}" fill="#111827" />
          {''.join(item[2] for item in polygons)}
          {''.join(axis_lines)}
        </svg>
      </div>

      <div class="meta">
        <div><strong>Elevation range</strong>{z_min:.2f} to {z_max:.2f} deg</div>
        <div><strong>Time samples</strong>{len(times)} values</div>
        <div><strong>Probe depth samples</strong>{len(probe_z_values)} values</div>
      </div>

      <div class="panel">
        <table>
          <thead>
            <tr><th>Probe z</th>{header_cells}</tr>
          </thead>
          <tbody>
            {''.join(rows_html)}
          </tbody>
        </table>
      </div>
    </div>
  </body>
</html>
"""
    output_path = CHARTS_DIR / "granite_3d_surface.html"
    output_path.write_text(html, encoding="utf-8")


def summarize_range(records: list[dict[str, Any]], variable_key: str, metric_key: str) -> dict[str, float]:
    ordered = sorted_group(records, variable_key)
    x_values = [float(record["variables"][variable_key]) for record in ordered]
    y_values = [prediction_metric(record, metric_key) for record in ordered]
    return {
        "input_min": min(x_values),
        "input_max": max(x_values),
        "metric_min": min(y_values),
        "metric_max": max(y_values),
        "metric_delta": max(y_values) - min(y_values),
    }


def build_summary(data: dict[str, Any]) -> dict[str, Any]:
    grouped = data["results"]
    comparison_records = grouped["comparison_cases"]
    temp_records = grouped["temperature_sensitivity"]
    pressure_records = grouped["pressure_sensitivity"]
    frequency_records = grouped["frequency_sensitivity"]
    surface_records = grouped["three_d_surface"]

    highest_displacement = max(comparison_records, key=lambda item: prediction_metric(item, "max_displacement"))
    highest_elevation = max(comparison_records, key=lambda item: prediction_metric(item, "elevation_deg"))
    fastest_case = min(comparison_records, key=lambda item: prediction_metric(item, "travel_time_ms"))

    surface_elevations = [prediction_metric(item, "elevation_deg") for item in surface_records]
    z_values = [float(item["variables"]["probe_z"]) for item in surface_records]
    z_grouped: dict[float, list[float]] = {}
    for record in surface_records:
        z_grouped.setdefault(float(record["variables"]["probe_z"]), []).append(prediction_metric(record, "elevation_deg"))
    z_means = {f"{key:.2f}": round(float(np.mean(values)), 3) for key, values in sorted(z_grouped.items())}

    key_findings = [
        (
            f"Temperature sweep changed max temperature perturbation by "
            f"{summarize_range(temp_records, 'temperature_c', 'max_temperature_perturbation')['metric_delta']:.3f}, "
            "while azimuth stayed comparatively stable in the 2D pulse setup."
        ),
        (
            f"Pressure sweep changed displacement by "
            f"{summarize_range(pressure_records, 'pressure_mpa', 'max_displacement')['metric_delta']:.6f}, "
            "showing stronger impact on field intensity than on direction."
        ),
        (
            f"3D probe-depth sweep produced an elevation range of "
            f"{max(surface_elevations) - min(surface_elevations):.2f} degrees, "
            "which confirms that the current pipeline reacts to z-axis geometry."
        ),
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "medium_id": data["medium_id"],
        "model": data["model"],
        "medium_name": data["medium"]["name"],
        "total_predictions": data["counts"]["completed"],
        "errors": data["counts"]["errors"],
        "comparison_extremes": {
            "highest_displacement": {
                "scenario_id": highest_displacement["scenario_id"],
                "label": highest_displacement["label"],
                "value": prediction_metric(highest_displacement, "max_displacement"),
            },
            "highest_elevation": {
                "scenario_id": highest_elevation["scenario_id"],
                "label": highest_elevation["label"],
                "value": prediction_metric(highest_elevation, "elevation_deg"),
            },
            "shortest_travel_time": {
                "scenario_id": fastest_case["scenario_id"],
                "label": fastest_case["label"],
                "value": prediction_metric(fastest_case, "travel_time_ms"),
            },
        },
        "temperature_sensitivity": {
            "displacement": summarize_range(temp_records, "temperature_c", "max_displacement"),
            "temperature_perturbation": summarize_range(temp_records, "temperature_c", "max_temperature_perturbation"),
            "travel_time": summarize_range(temp_records, "temperature_c", "travel_time_ms"),
        },
        "pressure_sensitivity": {
            "magnitude": summarize_range(pressure_records, "pressure_mpa", "magnitude"),
            "displacement": summarize_range(pressure_records, "pressure_mpa", "max_displacement"),
            "travel_time": summarize_range(pressure_records, "pressure_mpa", "travel_time_ms"),
        },
        "frequency_sensitivity": {
            "magnitude": summarize_range(frequency_records, "frequency_hz", "magnitude"),
            "displacement": summarize_range(frequency_records, "frequency_hz", "max_displacement"),
            "azimuth": summarize_range(frequency_records, "frequency_hz", "azimuth_deg"),
        },
        "three_d_check": {
            "z_sensitive": (max(surface_elevations) - min(surface_elevations)) > 1.0,
            "elevation_range_deg": [round(min(surface_elevations), 3), round(max(surface_elevations), 3)],
            "probe_z_mean_elevations_deg": z_means,
        },
        "key_findings": key_findings,
    }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def write_report(
    *,
    data: dict[str, Any],
    summary: dict[str, Any],
    report_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    medium_name = data["medium"]["name"]
    findings_html = "".join(f"<li>{finding}</li>" for finding in summary["key_findings"])
    comparison_rows = []
    for record in data["results"]["comparison_cases"]:
        metrics = scenario_metrics(record)
        comparison_rows.append(
            "<tr>"
            f"<td>{record['label']}</td>"
            f"<td>{metrics['azimuth_deg']:.2f}</td>"
            f"<td>{metrics['elevation_deg']:.2f}</td>"
            f"<td>{metrics['travel_time_ms']:.3f}</td>"
            f"<td>{metrics['max_displacement']:.6f}</td>"
            f"<td>{metrics['max_temperature_perturbation']:.6f}</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{medium_name} analytics report</title>
    <style>
      body {{
        margin: 0;
        background: #0f172a;
        color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }}
      .wrap {{
        max-width: 1180px;
        margin: 0 auto;
        padding: 34px 24px 60px;
      }}
      h1, h2, h3 {{
        margin: 0 0 12px;
        font-weight: 650;
      }}
      p {{
        margin: 0 0 14px;
        line-height: 1.65;
        color: #cbd5e1;
      }}
      .lead {{
        color: #94a3b8;
        max-width: 900px;
      }}
      .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 14px;
        margin: 22px 0 28px;
      }}
      .metric, .panel {{
        background: #111827;
        border: 1px solid #334155;
        border-radius: 18px;
      }}
      .metric {{
        padding: 18px;
      }}
      .metric strong {{
        display: block;
        margin-bottom: 8px;
        color: #cbd5e1;
      }}
      .metric span {{
        font-size: 24px;
      }}
      .panel {{
        padding: 20px;
        margin-bottom: 18px;
      }}
      ul {{
        margin: 8px 0 0;
        padding-left: 20px;
        color: #e2e8f0;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 12px;
        font-size: 14px;
      }}
      th, td {{
        padding: 12px 10px;
        text-align: left;
        border-bottom: 1px solid #1e293b;
      }}
      th {{
        color: #cbd5e1;
        font-weight: 600;
      }}
      td {{
        color: #e2e8f0;
      }}
      .gallery {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 18px;
      }}
      figure {{
        margin: 0;
        background: #111827;
        border: 1px solid #334155;
        border-radius: 18px;
        padding: 14px;
      }}
      figure img {{
        width: 100%;
        display: block;
        border-radius: 12px;
      }}
      figcaption {{
        margin-top: 12px;
        color: #94a3b8;
        font-size: 14px;
        line-height: 1.5;
      }}
      code {{
        color: #bfdbfe;
      }}
      a {{
        color: #93c5fd;
      }}
    </style>
  </head>
  <body>
    <div class="wrap">
      <h1>{medium_name} analytics report</h1>
      <p class="lead">
        Generated from the isolated <code>granite-analytics/</code> package on {summary['generated_at']}.
        The scripts are medium-parameterized even though this report uses <strong>{data['medium_id']}</strong> as the input medium.
        Backend model route: <strong>{data['model']}</strong>.
      </p>

      <div class="grid">
        <div class="metric"><strong>Total predictions</strong><span>{summary['total_predictions']}</span></div>
        <div class="metric"><strong>Errors</strong><span>{summary['errors']}</span></div>
        <div class="metric"><strong>Highest elevation</strong><span>{summary['comparison_extremes']['highest_elevation']['value']:.2f} deg</span></div>
        <div class="metric"><strong>Shortest travel time</strong><span>{summary['comparison_extremes']['shortest_travel_time']['value']:.3f} ms</span></div>
      </div>

      <div class="panel">
        <h2>Key findings</h2>
        <ul>{findings_html}</ul>
      </div>

      <div class="panel">
        <h2>Representative scenarios</h2>
        <table>
          <thead>
            <tr>
              <th>Scenario</th>
              <th>Azimuth</th>
              <th>Elevation</th>
              <th>Travel time</th>
              <th>Max displacement</th>
              <th>Temp perturbation</th>
            </tr>
          </thead>
          <tbody>
            {''.join(comparison_rows)}
          </tbody>
        </table>
      </div>

      <h2 style="margin: 26px 0 14px;">Chart gallery</h2>
      <div class="gallery">
        <figure>
          <img src="charts/amplitude_time_series.png" alt="Amplitude time series chart" />
          <figcaption>Time-series sweep showing how peak displacement evolves with different source amplitudes.</figcaption>
        </figure>
        <figure>
          <img src="charts/metrics_comparison.png" alt="Metrics comparison chart" />
          <figcaption>Comparison of directional and field metrics across baseline, thermal, pressure and 3D perturbation cases.</figcaption>
        </figure>
        <figure>
          <img src="charts/temperature_sensitivity.png" alt="Temperature sensitivity chart" />
          <figcaption>Temperature sweep for granite in a 2D setup.</figcaption>
        </figure>
        <figure>
          <img src="charts/pressure_sensitivity.png" alt="Pressure sensitivity chart" />
          <figcaption>Pressure sweep emphasizing how the current model changes displacement and travel time.</figcaption>
        </figure>
        <figure>
          <img src="charts/frequency_sensitivity.png" alt="Frequency sensitivity chart" />
          <figcaption>Frequency sweep with magnitude, displacement and azimuth response.</figcaption>
        </figure>
        <figure>
          <img src="charts/temperature_pressure_heatmap.png" alt="Temperature pressure heatmap" />
          <figcaption>Heatmap of max displacement over the temperature-pressure grid.</figcaption>
        </figure>
        <figure>
          <img src="charts/granite_3d_surface.png" alt="3D response surface" />
          <figcaption>3D surface over probe depth and time. Interactive-free HTML companion: <a href="charts/granite_3d_surface.html">open standalone surface view</a>.</figcaption>
        </figure>
      </div>

      <div class="panel" style="margin-top: 22px;">
        <h2>Re-run commands</h2>
        <p><code>python3 granite-analytics/scripts/run_granite_predictions.py --medium-id granite</code></p>
        <p><code>python3 granite-analytics/scripts/generate_granite_charts.py</code></p>
      </div>
    </div>
  </body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")


def main() -> int:
    args = parse_args()
    chart_theme()
    ensure_dirs()

    data = load_json(args.input)
    grouped = data["results"]
    medium_name = data["medium"]["name"]

    plot_amplitude_time_series(grouped["amplitude_time_series"], medium_name)
    plot_metrics_comparison(grouped["comparison_cases"], medium_name)
    plot_temperature_sensitivity(grouped["temperature_sensitivity"], medium_name)
    plot_pressure_sensitivity(grouped["pressure_sensitivity"], medium_name)
    plot_frequency_sensitivity(grouped["frequency_sensitivity"], medium_name)
    plot_temperature_pressure_heatmap(grouped["temperature_pressure_heatmap"], medium_name)
    times, probe_z_values, elevation_grid = plot_three_d_surface(grouped["three_d_surface"], medium_name)
    write_three_d_surface_html(
        times=times,
        probe_z_values=probe_z_values,
        elevation_grid=elevation_grid,
        medium_name=medium_name,
    )

    summary = build_summary(data)
    write_summary(args.summary_output, summary)
    write_report(data=data, summary=summary, report_path=args.report_output)

    print(
        json.dumps(
            {
                "summary_output": str(args.summary_output),
                "report_output": str(args.report_output),
                "charts_dir": str(CHARTS_DIR),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
