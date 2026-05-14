#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from analyze_model_outputs import (
    AnalysisResult,
    MODEL_ORDER,
    circular_distance_deg,
    circular_std_deg,
    filter_rows,
    load_and_analyze_summary,
    material_sort_key,
    model_sort_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate comparison charts from normalized model-service experiment results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/data_experiments/results/summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/data_experiments/charts"),
    )
    parser.add_argument(
        "--include-fallback",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Include fallback responses in scientific comparison plots.",
    )
    return parser.parse_args()


def float_or_none(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def render_placeholder(plt, output_path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def sanitize_heatmap_value(value: float | None, *, log_scale: bool) -> float | None:
    if value is None:
        return None
    if log_scale:
        if value <= 0:
            return None
        return math.log10(value)
    return value


def render_heatmap(
    plt,
    *,
    output_path: Path,
    title: str,
    x_labels: list[str],
    y_labels: list[str],
    values: list[list[float | None]],
    colorbar_label: str,
    cell_formatter,
    cmap: str = "viridis",
) -> None:
    if not x_labels or not y_labels or not values:
        render_placeholder(plt, output_path, title, "Not enough data to render this heatmap.")
        return

    flat_values = [value for row in values for value in row if value is not None]
    if not flat_values:
        render_placeholder(plt, output_path, title, "All values are empty for this heatmap.")
        return

    fill_value = min(flat_values)
    masked = [[fill_value if value is None else value for value in row] for row in values]

    fig_width = max(8, len(x_labels) * 1.3)
    fig_height = max(5, len(y_labels) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(masked, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)

    threshold = (min(flat_values) + max(flat_values)) / 2.0
    for row_index, row_values in enumerate(values):
        for col_index, value in enumerate(row_values):
            text = "NA" if value is None else cell_formatter(value)
            text_color = "white" if value is not None and value >= threshold else "black"
            ax.text(col_index, row_index, text, ha="center", va="center", fontsize=7, color=text_color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def average_metric(rows: list[dict[str, str]], metric: str, *, by: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        key = row.get(by, "")
        value = float_or_none(row.get(metric, ""))
        if key and value is not None:
            buckets.setdefault(key, []).append(value)
    return {key: sum(values) / len(values) for key, values in buckets.items() if values}


def apply_log_scale_if_needed(ax, values: list[float]) -> None:
    positive = [value for value in values if value > 0]
    if len(positive) < 2:
        return
    spread = max(positive) / max(min(positive), 1e-12)
    if spread >= 100:
        ax.set_yscale("log")
        ax.set_ylabel(f"{ax.get_ylabel()} (log scale)")


def annotate_bars(ax, values: list[float]) -> None:
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.3g}", ha="center", va="bottom", fontsize=8, rotation=0)


def canonical_model_labels(rows: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for model in MODEL_ORDER:
        candidates = [row for row in rows if row.get("model") == model]
        if not candidates:
            continue
        if all(row.get("is_fallback") for row in candidates):
            labels.append(f"{model} (fallback)")
        else:
            labels.append(model)
    return labels


def plot_bar_with_missing_annotations(
    plt,
    *,
    output_path: Path,
    title: str,
    y_label: str,
    labels: list[str],
    values_map: dict[str, float],
    color: str,
    missing_reason: str,
    log_scale: bool = False,
) -> None:
    if not labels:
        render_placeholder(plt, output_path, title, "No model labels were available.")
        return
    values = [values_map.get(label, 0.0) for label in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    if log_scale and any(value > 0 for value in values):
        ax.set_yscale("log")
    floor = max((min(value for value in values if value > 0) * 0.5), 1e-6) if any(value > 0 for value in values) else 1e-6
    if log_scale:
        ax.set_ylim(bottom=floor)
    for bar, label, value in zip(bars, labels, values):
        if label in values_map:
            ax.text(bar.get_x() + bar.get_width() / 2, value if value > 0 else floor, f"{value:.3g}", ha="center", va="bottom", fontsize=8)
        else:
            bar.set_facecolor("#CBD5E1")
            bar.set_edgecolor("#475569")
            bar.set_hatch("//")
            ax.text(bar.get_x() + bar.get_width() / 2, floor, missing_reason, ha="center", va="bottom", fontsize=8, rotation=90, color="#334155")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def grouped_mean(
    rows: list[dict[str, str]],
    *,
    group_keys: tuple[str, ...],
    metric: str,
) -> dict[tuple[str, ...], float]:
    buckets: dict[tuple[str, ...], list[float]] = {}
    for row in rows:
        value = float_or_none(row.get(metric, ""))
        if value is None:
            continue
        key = tuple(row.get(name, "") for name in group_keys)
        if all(key):
            buckets.setdefault(key, []).append(value)
    return {key: sum(values) / len(values) for key, values in buckets.items() if values}


def plot_temperature_comparison(plt, rows: list[dict[str, str]], output_path: Path) -> None:
    means = average_metric(rows, "max_temperature_perturbation", by="model")
    labels = [label.split(" ")[0] for label in canonical_model_labels(rows)]
    if not means and not labels:
        render_placeholder(plt, output_path, "Temperature comparison", "No successful temperature metrics were available.")
        return
    plot_bar_with_missing_annotations(
        plt,
        output_path=output_path,
        title="Temperature perturbation comparison",
        y_label="Mean max temperature perturbation",
        labels=labels,
        values_map=means,
        color="#4F46E5",
        missing_reason="excluded",
        log_scale=True,
    )


def plot_direction_components_proxy(plt, rows: list[dict[str, str]], output_path: Path) -> None:
    x_means = average_metric(rows, "direction_x", by="model")
    y_means = average_metric(rows, "direction_y", by="model")
    z_means = average_metric(rows, "direction_z", by="model")
    if not x_means and not y_means and not z_means:
        render_placeholder(
            plt,
            output_path,
            "Direction components comparison",
            "No normalized direction components were available.",
        )
        return
    labels = sorted(set(x_means) | set(y_means) | set(z_means))
    x_values = [x_means.get(label, 0.0) for label in labels]
    y_values = [y_means.get(label, 0.0) for label in labels]
    z_values = [z_means.get(label, 0.0) for label in labels]
    indices = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(11, 5))
    if all(abs(value) < 1e-9 for value in z_values):
        width = 0.35
        ax.bar([index - width / 2 for index in indices], x_values, width=width, label="direction_x")
        ax.bar([index + width / 2 for index in indices], y_values, width=width, label="direction_y")
        ax.set_title("Direction-component comparison for 2D runs (x, y only)")
    else:
        width = 0.25
        ax.bar([index - width for index in indices], x_values, width=width, label="direction_x")
        ax.bar(indices, y_values, width=width, label="direction_y")
        ax.bar([index + width for index in indices], z_values, width=width, label="direction_z")
        ax.set_title("Direction-component comparison")
    ax.set_xticks(indices)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_displacement_magnitude(plt, rows: list[dict[str, str]], output_path: Path) -> None:
    means = average_metric(rows, "max_displacement", by="model")
    labels = [label.split(" ")[0] for label in canonical_model_labels(rows)]
    if not means and not labels:
        render_placeholder(plt, output_path, "Displacement magnitude comparison", "No displacement metrics were available.")
        return
    plot_bar_with_missing_annotations(
        plt,
        output_path=output_path,
        title="Max displacement comparison (successful responses only)",
        y_label="Mean max displacement",
        labels=labels,
        values_map=means,
        color="#0F766E",
        missing_reason="excluded",
        log_scale=True,
    )


def plot_displacement_valid_only(plt, rows: list[dict[str, Any]], output_path: Path) -> None:
    valid_rows = [row for row in rows if not row["is_outlier"]]
    means = average_metric(valid_rows, "max_displacement", by="model_label")
    labels = canonical_model_labels(rows)
    if not means and not labels:
        render_placeholder(plt, output_path, "Max displacement (valid only)", "No valid non-outlier displacement rows were available.")
        return
    plot_bar_with_missing_annotations(
        plt,
        output_path=output_path,
        title="Max displacement (valid non-outlier responses)",
        y_label="Mean max displacement",
        labels=labels,
        values_map=means,
        color="#0F766E",
        missing_reason="outlier\nexcluded",
    )


def plot_displacement_log_diagnostic(plt, rows: list[dict[str, Any]], output_path: Path) -> None:
    means = average_metric(rows, "max_displacement", by="model_label")
    labels = canonical_model_labels(rows)
    if not means and not labels:
        render_placeholder(plt, output_path, "Max displacement log diagnostic", "No displacement rows were available.")
        return
    plot_bar_with_missing_annotations(
        plt,
        output_path=output_path,
        title="Max displacement log diagnostic",
        y_label="Mean max displacement",
        labels=labels,
        values_map=means,
        color="#0F766E",
        missing_reason="no data",
        log_scale=True,
    )


def plot_temperature_valid_only(plt, rows: list[dict[str, Any]], output_path: Path) -> None:
    valid_rows = [row for row in rows if not row["is_outlier"]]
    means = average_metric(valid_rows, "max_temperature_perturbation", by="model_label")
    labels = canonical_model_labels(rows)
    if not means and not labels:
        render_placeholder(plt, output_path, "Temperature perturbation (valid only)", "No valid non-outlier temperature rows were available.")
        return
    plot_bar_with_missing_annotations(
        plt,
        output_path=output_path,
        title="Temperature perturbation (valid non-outlier responses)",
        y_label="Mean max temperature perturbation",
        labels=labels,
        values_map=means,
        color="#4F46E5",
        missing_reason="outlier\nexcluded",
    )


def plot_temperature_log_diagnostic(plt, rows: list[dict[str, Any]], output_path: Path) -> None:
    means = average_metric(rows, "max_temperature_perturbation", by="model_label")
    labels = canonical_model_labels(rows)
    if not means and not labels:
        render_placeholder(plt, output_path, "Temperature perturbation log diagnostic", "No temperature rows were available.")
        return
    plot_bar_with_missing_annotations(
        plt,
        output_path=output_path,
        title="Temperature perturbation log diagnostic",
        y_label="Mean max temperature perturbation",
        labels=labels,
        values_map=means,
        color="#4F46E5",
        missing_reason="no data",
        log_scale=True,
    )


def plot_material_comparison(plt, rows: list[dict[str, str]], output_path: Path) -> None:
    materials = sorted({row["material"] for row in rows if row.get("material")}, key=material_sort_key)
    models = canonical_model_labels(rows)
    if len(materials) < 2:
        render_placeholder(plt, output_path, "Material comparison", "Need at least two materials with successful results.")
        return
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = float_or_none(row.get("travel_time_ms_pred", ""))
        if value is not None:
            buckets[(row["material"], row["model_label"])].append(value)
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.18
    base = list(range(len(materials)))
    for idx, model in enumerate(models):
        values = []
        for material in materials:
            samples = buckets.get((material, model), [])
            values.append(sum(samples) / len(samples) if samples else 0.0)
        offset = (idx - (len(models) - 1) / 2.0) * width
        ax.bar([x + offset for x in base], values, width=width, label=model)
    ax.set_xticks(base)
    ax.set_xticklabels(materials)
    ax.set_title("Travel-time comparison by material")
    ax.set_ylabel("Mean predicted travel time (ms)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_model_disagreement(plt, rows: list[dict[str, str]], output_path: Path) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["case_id"]].append(row)
    case_ids: list[str] = []
    disagreements: list[float] = []
    for case_id, items in grouped.items():
        values = [float_or_none(item.get("azimuth_deg", "")) for item in items]
        values = [value for value in values if value is not None]
        if len(values) >= 2:
            variance = circular_std_deg(values)
            case_ids.append(case_id)
            disagreements.append(float(variance or 0.0))
    if not disagreements:
        render_placeholder(plt, output_path, "Model disagreement", "Need at least two successful model outputs per case.")
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(case_ids, disagreements, color="#B45309")
    ax.set_title("Azimuth circular disagreement across models")
    ax.set_ylabel("Circular std. of azimuth (deg)")
    ax.tick_params(axis="x", rotation=75)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_prediction_vs_time(plt, rows: list[dict[str, str]], output_path: Path) -> None:
    grouped: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    for row in rows:
        time_value = float_or_none(row.get("time_ms", ""))
        travel_time = float_or_none(row.get("travel_time_ms_pred", ""))
        max_displacement = float_or_none(row.get("max_displacement", ""))
        if time_value is not None and travel_time is not None and max_displacement is not None:
            grouped[(row["material"], row["model_label"])].append((time_value, travel_time, max_displacement))
    enough_variation = any(len({time for time, *_ in points}) >= 2 for points in grouped.values())
    if not enough_variation:
        render_placeholder(
            plt,
            output_path,
            "Prediction vs time",
            "Current experiment input pack uses a mostly fixed time point. Generate a multi-time grid to produce this chart.",
        )
        return
    materials = sorted({material for material, _ in grouped}, key=material_sort_key)
    fig, axes = plt.subplots(len(materials), 2, figsize=(14, 5 * len(materials)), squeeze=False)
    for row_index, material in enumerate(materials):
        travel_ax = axes[row_index][0]
        disp_ax = axes[row_index][1]
        for _, model in sorted((key for key in grouped if key[0] == material), key=lambda item: model_sort_key(item[1].split(" ")[0])):
            points = sorted(grouped[(material, model)])
            travel_ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                label=model,
            )
            disp_ax.plot(
                [point[0] for point in points],
                [point[2] for point in points],
                marker="o",
                label=model,
            )
        travel_ax.set_title(f"{material}: travel time vs time")
        travel_ax.set_xlabel("Scenario time (ms)")
        travel_ax.set_ylabel("Predicted travel time (ms)")
        travel_ax.legend()
        disp_ax.set_title(f"{material}: displacement vs time")
        disp_ax.set_xlabel("Scenario time (ms)")
        disp_ax.set_ylabel("Max displacement")
        disp_ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_basalt_vs_sandstone_travel_time(plt, rows: list[dict[str, str]], output_path: Path) -> None:
    means = grouped_mean(
        rows,
        group_keys=("material", "model"),
        metric="travel_time_ms_pred",
    )
    if not means:
        render_placeholder(plt, output_path, "Basalt vs sandstone travel time", "No travel-time metrics were available.")
        return
    materials = [material for material in ("basalt", "sandstone") if any(key[0] == material for key in means)]
    models = canonical_model_labels(rows)
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.18
    base = list(range(len(materials)))
    for idx, model in enumerate(models):
        values = [means.get((material, model), 0.0) for material in materials]
        offset = (idx - (len(models) - 1) / 2.0) * width
        ax.bar([x + offset for x in base], values, width=width, label=model)
    ax.set_xticks(base)
    ax.set_xticklabels(materials)
    ax.set_title("Basalt vs sandstone: predicted travel time")
    ax.set_ylabel("Mean predicted travel time (ms)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_basalt_vs_sandstone_displacement(plt, rows: list[dict[str, str]], output_path: Path) -> None:
    means = grouped_mean(
        rows,
        group_keys=("material", "model"),
        metric="max_displacement",
    )
    if not means:
        render_placeholder(plt, output_path, "Basalt vs sandstone displacement", "No displacement metrics were available.")
        return
    materials = [material for material in ("basalt", "sandstone") if any(key[0] == material for key in means)]
    models = canonical_model_labels(rows)
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.18
    base = list(range(len(materials)))
    for idx, model in enumerate(models):
        values = [means.get((material, model), 0.0) for material in materials]
        offset = (idx - (len(models) - 1) / 2.0) * width
        ax.bar([x + offset for x in base], values, width=width, label=model)
    ax.set_xticks(base)
    ax.set_xticklabels(materials)
    ax.set_title("Basalt vs sandstone: max displacement")
    ax.set_ylabel("Mean max displacement")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_elevation_comparison(plt, rows: list[dict[str, str]], output_path: Path) -> None:
    means = average_metric(rows, "elevation_deg", by="model")
    labels = [label.split(" ")[0] for label in canonical_model_labels(rows)]
    if not means and not labels:
        render_placeholder(plt, output_path, "Elevation comparison", "No elevation metrics were available.")
        return
    values = [means.get(label, 0.0) for label in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color="#7C3AED")
    ax.set_title("Mean elevation angle by model")
    ax.set_ylabel("Elevation (deg)")
    for bar, label, value in zip(bars, labels, values):
        note = " (2D-adapted)" if label == "fno" and abs(value) < 1e-9 else ""
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3g}{note}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_depth_sensitivity(plt, rows: list[dict[str, str]], output_path: Path) -> None:
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        probe_z = float_or_none(row.get("probe_z", ""))
        travel_time = float_or_none(row.get("travel_time_ms_pred", ""))
        if probe_z is not None and travel_time is not None:
            grouped[(row["material"], row["model"])].append((probe_z, travel_time))
    enough_variation = any(len({z for z, _ in points}) >= 2 for points in grouped.values())
    if not enough_variation:
        render_placeholder(
            plt,
            output_path,
            "Depth sensitivity",
            "Need at least two distinct probe_z values per model/material to show depth sensitivity.",
        )
        return
    materials = sorted({material for material, _ in grouped})
    fig, axes = plt.subplots(len(materials), 1, figsize=(12, 4.5 * len(materials)), squeeze=False)
    for row_index, material in enumerate(materials):
        ax = axes[row_index][0]
        for _, model in sorted(key for key in grouped if key[0] == material):
            points = sorted(grouped[(material, model)])
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                label=model,
            )
        ax.set_title(f"{material}: travel time vs probe depth")
        ax.set_xlabel("Probe z")
        ax.set_ylabel("Predicted travel time (ms)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_depth_metric(
    plt,
    rows: list[dict[str, str]],
    output_path: Path,
    *,
    metric_key: str,
    title: str,
    y_label: str,
) -> None:
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        probe_z = float_or_none(row.get("probe_z", ""))
        metric_value = float_or_none(row.get(metric_key, ""))
        if probe_z is not None and metric_value is not None:
            grouped[(row["material"], row["model_label"])].append((probe_z, metric_value))
    enough_variation = any(len({z for z, _ in points}) >= 2 for points in grouped.values())
    if not enough_variation:
        render_placeholder(
            plt,
            output_path,
            title,
            "Need at least two distinct probe_z values per model/material to show depth sensitivity.",
        )
        return
    materials = sorted({material for material, _ in grouped}, key=material_sort_key)
    fig, axes = plt.subplots(len(materials), 1, figsize=(12, 4.5 * len(materials)), squeeze=False)
    limited = True
    for row_index, material in enumerate(materials):
        ax = axes[row_index][0]
        for _, model in sorted(
            (key for key in grouped if key[0] == material),
            key=lambda item: model_sort_key(item[1].split(" ")[0]),
        ):
            points = sorted(grouped[(material, model)])
            if len({z for z, _ in points}) > 2:
                limited = False
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                label=model,
            )
        suffix = " (limited diagnostic: only 2 depth samples)" if limited else ""
        ax.set_title(f"{material}: {title.lower()}{suffix}")
        ax.set_xlabel("Probe z")
        ax.set_ylabel(y_label)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_domain_adaptation_summary(plt, rows: list[dict[str, str]], output_path: Path) -> None:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        model = row.get("model", "")
        adaptation = row.get("domain_adaptation", "") or "none"
        if model:
            counts[model][adaptation] += 1
    if not counts:
        render_placeholder(plt, output_path, "Domain adaptation summary", "No rows were available in summary.csv.")
        return
    labels = sorted(counts.keys())
    adaptation_labels = sorted({adaptation for counter in counts.values() for adaptation in counter.keys()})
    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = [0] * len(labels)
    palette = ["#15803D", "#B45309", "#1D4ED8", "#B91C1C", "#7C3AED"]
    for index, adaptation in enumerate(adaptation_labels):
        values = [counts[label].get(adaptation, 0) for label in labels]
        ax.bar(labels, values, bottom=bottom, label=adaptation, color=palette[index % len(palette)])
        bottom = [current + delta for current, delta in zip(bottom, values)]
    ax.set_title("Requested vs effective domain adaptation")
    ax.set_ylabel("Case count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_case_model_heatmap(
    plt,
    rows: list[dict[str, str]],
    output_path: Path,
    *,
    metric_key: str,
    title: str,
    colorbar_label: str,
    log_scale: bool = False,
) -> None:
    case_ids = sorted({row["case_id"] for row in rows})
    model_labels = sorted(
        {row["model_label"] for row in rows},
        key=lambda label: model_sort_key(label.split(" ")[0]),
    )
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        metric_value = float_or_none(row.get(metric_key, ""))
        if metric_value is not None:
            grouped[(row["case_id"], row["model_label"])].append(metric_value)
    values: list[list[float | None]] = []
    for case_id in case_ids:
        row_values: list[float | None] = []
        for model_label in model_labels:
            samples = grouped.get((case_id, model_label), [])
            value = sum(samples) / len(samples) if samples else None
            row_values.append(sanitize_heatmap_value(value, log_scale=log_scale))
        values.append(row_values)
    render_heatmap(
        plt,
        output_path=output_path,
        title=title,
        x_labels=model_labels,
        y_labels=case_ids,
        values=values,
        colorbar_label=colorbar_label,
        cell_formatter=lambda value: f"{value:.2f}" if log_scale else f"{value:.3g}",
        cmap="magma" if log_scale else "viridis",
    )


def plot_pairwise_disagreement_heatmap(
    plt,
    rows: list[dict[str, str]],
    output_path: Path,
    *,
    title: str,
    metric_getter,
    distance_fn,
    colorbar_label: str,
) -> None:
    model_labels = sorted(
        {row["model_label"] for row in rows},
        key=lambda label: model_sort_key(label.split(" ")[0]),
    )
    by_case_model = {(row["case_id"], row["model_label"]): row for row in rows}
    case_ids = sorted({row["case_id"] for row in rows})
    values: list[list[float | None]] = []
    for model_a in model_labels:
        matrix_row: list[float | None] = []
        for model_b in model_labels:
            diffs: list[float] = []
            for case_id in case_ids:
                row_a = by_case_model.get((case_id, model_a))
                row_b = by_case_model.get((case_id, model_b))
                if not row_a or not row_b:
                    continue
                value_a = metric_getter(row_a)
                value_b = metric_getter(row_b)
                if value_a is None or value_b is None:
                    continue
                diffs.append(distance_fn(value_a, value_b))
            matrix_row.append(sum(diffs) / len(diffs) if diffs else None)
        values.append(matrix_row)
    render_heatmap(
        plt,
        output_path=output_path,
        title=title,
        x_labels=model_labels,
        y_labels=model_labels,
        values=values,
        colorbar_label=colorbar_label,
        cell_formatter=lambda value: f"{value:.2f}",
        cmap="cividis",
    )


def plot_material_model_heatmap(
    plt,
    rows: list[dict[str, str]],
    output_path: Path,
    *,
    metric_key: str,
    title: str,
    colorbar_label: str,
    log_scale: bool = False,
) -> None:
    materials = sorted({row["material"] for row in rows}, key=material_sort_key)
    model_labels = sorted(
        {row["model_label"] for row in rows},
        key=lambda label: model_sort_key(label.split(" ")[0]),
    )
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        metric_value = float_or_none(row.get(metric_key, ""))
        if metric_value is not None:
            grouped[(row["material"], row["model_label"])].append(metric_value)
    values: list[list[float | None]] = []
    for material in materials:
        row_values: list[float | None] = []
        for model_label in model_labels:
            samples = grouped.get((material, model_label), [])
            value = sum(samples) / len(samples) if samples else None
            row_values.append(sanitize_heatmap_value(value, log_scale=log_scale))
        values.append(row_values)
    render_heatmap(
        plt,
        output_path=output_path,
        title=title,
        x_labels=model_labels,
        y_labels=materials,
        values=values,
        colorbar_label=colorbar_label,
        cell_formatter=lambda value: f"{value:.2f}" if log_scale else f"{value:.3g}",
        cmap="magma" if log_scale else "viridis",
    )


def plot_parameter_model_heatmap(
    plt,
    rows: list[dict[str, str]],
    output_path: Path,
    *,
    parameter_key: str,
    metric_key: str,
    title: str,
    colorbar_label: str,
    log_scale: bool = False,
) -> bool:
    parameter_values = sorted(
        {
            float_or_none(row.get(parameter_key, ""))
            for row in rows
            if float_or_none(row.get(parameter_key, "")) is not None
        }
    )
    parameter_values = [value for value in parameter_values if value is not None]
    if len(parameter_values) < 2:
        return False
    model_labels = sorted(
        {row["model_label"] for row in rows},
        key=lambda label: model_sort_key(label.split(" ")[0]),
    )
    grouped: dict[tuple[float, str], list[float]] = defaultdict(list)
    for row in rows:
        parameter_value = float_or_none(row.get(parameter_key, ""))
        metric_value = float_or_none(row.get(metric_key, ""))
        if parameter_value is not None and metric_value is not None:
            grouped[(parameter_value, row["model_label"])].append(metric_value)
    values: list[list[float | None]] = []
    for parameter_value in parameter_values:
        row_values: list[float | None] = []
        for model_label in model_labels:
            samples = grouped.get((parameter_value, model_label), [])
            value = sum(samples) / len(samples) if samples else None
            row_values.append(sanitize_heatmap_value(value, log_scale=log_scale))
        values.append(row_values)
    render_heatmap(
        plt,
        output_path=output_path,
        title=title,
        x_labels=model_labels,
        y_labels=[f"{value:g}" for value in parameter_values],
        values=values,
        colorbar_label=colorbar_label,
        cell_formatter=lambda value: f"{value:.2f}" if log_scale else f"{value:.3g}",
        cmap="magma" if log_scale else "viridis",
    )
    return True


def plot_model_validity_summary(plt, analysis: AnalysisResult, output_path: Path) -> None:
    by_model = analysis.stats["by_model"]
    if not by_model:
        render_placeholder(plt, output_path, "Model validity summary", "No model stats were available.")
        return
    labels = sorted(by_model.keys(), key=model_sort_key)
    checkpoint_values = [by_model[label].get("ok_checkpoint", 0) for label in labels]
    fallback_values = [by_model[label].get("ok_fallback", 0) for label in labels]
    error_values = [by_model[label].get("error", 0) for label in labels]
    outlier_values = [
        sum(1 for row in analysis.rows if row["model"] == label and row["is_outlier"])
        for label in labels
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(labels, checkpoint_values, label="checkpoint", color="#15803D")
    bottom = checkpoint_values[:]
    ax.bar(labels, fallback_values, bottom=bottom, label="fallback", color="#B45309")
    bottom = [a + b for a, b in zip(bottom, fallback_values)]
    ax.bar(labels, error_values, bottom=bottom, label="error", color="#B91C1C")
    bottom = [a + b for a, b in zip(bottom, error_values)]
    ax.bar(labels, outlier_values, bottom=bottom, label="outlier", color="#7C3AED")
    ax.set_title("Model validity summary")
    ax.set_ylabel("Response count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_service_status_summary(plt, rows: list[dict[str, str]], output_path: Path) -> None:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        model = row.get("model", "")
        if not model:
            continue
        if row.get("status") == "ok" and row.get("fallback_used") == "True":
            counts[model]["ok_fallback"] += 1
        elif row.get("status") == "ok" and row.get("service_mode") == "checkpoint":
            counts[model]["ok_checkpoint"] += 1
        elif row.get("status") == "error" and "timed out" in str(row.get("error_message", "")).lower():
            counts[model]["timeout"] += 1
        elif row.get("status") == "error":
            counts[model]["error"] += 1
    if not counts:
        render_placeholder(plt, output_path, "Service status summary", "No rows were available in summary.csv.")
        return
    labels = sorted(counts.keys(), key=model_sort_key)
    checkpoint_values = [counts[label].get("ok_checkpoint", 0) for label in labels]
    fallback_values = [counts[label].get("ok_fallback", 0) for label in labels]
    error_values = [counts[label].get("error", 0) for label in labels]
    timeout_values = [counts[label].get("timeout", 0) for label in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, checkpoint_values, label="ok_checkpoint", color="#15803D")
    ax.bar(labels, fallback_values, bottom=checkpoint_values, label="ok_fallback", color="#B45309")
    error_bottom = [ok + fallback for ok, fallback in zip(checkpoint_values, fallback_values)]
    ax.bar(labels, error_values, bottom=error_bottom, label="error", color="#B91C1C")
    timeout_bottom = [a + b for a, b in zip(error_bottom, error_values)]
    ax.bar(labels, timeout_values, bottom=timeout_bottom, label="timeout", color="#1D4ED8")
    ax.set_title("Service status summary")
    ax.set_ylabel("Case count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    include_fallback = args.include_fallback == "true"
    analysis = load_and_analyze_summary(args.input)
    rows = analysis.rows
    scientific_rows = filter_rows(rows, include_fallback=include_fallback, only_ok=True, exclude_outliers=False)
    valid_rows = filter_rows(rows, include_fallback=include_fallback, only_ok=True, exclude_outliers=True)
    ensure_output_dir(args.output_dir)
    cache_dir = args.output_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir = args.output_dir / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir.resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir.resolve()))

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required to generate charts. Install it in your active environment first."
        ) from exc

    plot_temperature_comparison(
        plt,
        scientific_rows,
        args.output_dir / "temperature_comparison.png",
    )
    plot_direction_components_proxy(
        plt,
        scientific_rows,
        args.output_dir / "displacement_components_comparison.png",
    )
    plot_displacement_magnitude(
        plt,
        scientific_rows,
        args.output_dir / "displacement_magnitude_comparison.png",
    )
    plot_displacement_valid_only(
        plt,
        valid_rows,
        args.output_dir / "max_displacement_valid_only.png",
    )
    plot_displacement_log_diagnostic(
        plt,
        scientific_rows,
        args.output_dir / "max_displacement_log_diagnostic.png",
    )
    plot_temperature_valid_only(
        plt,
        valid_rows,
        args.output_dir / "temperature_perturbation_valid_only.png",
    )
    plot_temperature_log_diagnostic(
        plt,
        scientific_rows,
        args.output_dir / "temperature_perturbation_log_diagnostic.png",
    )
    plot_material_comparison(
        plt,
        scientific_rows,
        args.output_dir / "material_comparison_sandstone_vs_basalt.png",
    )
    plot_model_disagreement(
        plt,
        scientific_rows,
        args.output_dir / "azimuth_circular_disagreement_by_case.png",
    )
    plot_prediction_vs_time(
        plt,
        scientific_rows,
        args.output_dir / "prediction_vs_time.png",
    )
    plot_service_status_summary(
        plt,
        rows,
        args.output_dir / "service_status_summary.png",
    )
    plot_model_validity_summary(
        plt,
        analysis,
        args.output_dir / "model_validity_summary.png",
    )
    plot_basalt_vs_sandstone_travel_time(
        plt,
        scientific_rows,
        args.output_dir / "basalt_vs_sandstone_travel_time.png",
    )
    plot_basalt_vs_sandstone_displacement(
        plt,
        scientific_rows,
        args.output_dir / "basalt_vs_sandstone_displacement.png",
    )
    plot_elevation_comparison(
        plt,
        scientific_rows,
        args.output_dir / "elevation_comparison.png",
    )
    plot_depth_sensitivity(
        plt,
        scientific_rows,
        args.output_dir / "depth_sensitivity.png",
    )
    plot_depth_metric(
        plt,
        scientific_rows,
        args.output_dir / "depth_sensitivity_travel_time.png",
        metric_key="travel_time_ms_pred",
        title="Travel time vs probe depth",
        y_label="Predicted travel time (ms)",
    )
    plot_depth_metric(
        plt,
        scientific_rows,
        args.output_dir / "depth_sensitivity_displacement.png",
        metric_key="max_displacement",
        title="Displacement vs probe depth",
        y_label="Max displacement",
    )
    plot_depth_metric(
        plt,
        scientific_rows,
        args.output_dir / "depth_sensitivity_temperature.png",
        metric_key="max_temperature_perturbation",
        title="Temperature perturbation vs probe depth",
        y_label="Max temperature perturbation",
    )
    plot_domain_adaptation_summary(
        plt,
        rows,
        args.output_dir / "domain_adaptation_summary.png",
    )
    plot_case_model_heatmap(
        plt,
        scientific_rows,
        args.output_dir / "heatmap_case_model_travel_time.png",
        metric_key="travel_time_ms_pred",
        title="Case × model heatmap: travel time",
        colorbar_label="Travel time (ms)",
    )
    plot_case_model_heatmap(
        plt,
        scientific_rows,
        args.output_dir / "heatmap_case_model_displacement.png",
        metric_key="max_displacement",
        title="Case × model heatmap: log10(max displacement)",
        colorbar_label="log10(max displacement)",
        log_scale=True,
    )
    plot_case_model_heatmap(
        plt,
        scientific_rows,
        args.output_dir / "heatmap_case_model_temperature.png",
        metric_key="max_temperature_perturbation",
        title="Case × model heatmap: log10(max temperature perturbation)",
        colorbar_label="log10(max temperature perturbation)",
        log_scale=True,
    )
    plot_pairwise_disagreement_heatmap(
        plt,
        scientific_rows,
        args.output_dir / "heatmap_model_disagreement_travel_time.png",
        title="Model disagreement heatmap: travel time",
        metric_getter=lambda row: float_or_none(row.get("travel_time_ms_pred")),
        distance_fn=lambda a, b: abs(a - b),
        colorbar_label="Mean absolute difference (ms)",
    )
    plot_pairwise_disagreement_heatmap(
        plt,
        scientific_rows,
        args.output_dir / "heatmap_model_disagreement_displacement.png",
        title="Model disagreement heatmap: max displacement",
        metric_getter=lambda row: float_or_none(row.get("max_displacement")),
        distance_fn=lambda a, b: abs(a - b),
        colorbar_label="Mean absolute difference",
    )
    plot_pairwise_disagreement_heatmap(
        plt,
        scientific_rows,
        args.output_dir / "heatmap_model_disagreement_temperature.png",
        title="Model disagreement heatmap: temperature perturbation",
        metric_getter=lambda row: float_or_none(row.get("max_temperature_perturbation")),
        distance_fn=lambda a, b: abs(a - b),
        colorbar_label="Mean absolute difference",
    )
    plot_pairwise_disagreement_heatmap(
        plt,
        scientific_rows,
        args.output_dir / "heatmap_model_disagreement_azimuth.png",
        title="Model disagreement heatmap: azimuth",
        metric_getter=lambda row: float_or_none(row.get("azimuth_deg")),
        distance_fn=circular_distance_deg,
        colorbar_label="Circular mean absolute difference (deg)",
    )
    plot_pairwise_disagreement_heatmap(
        plt,
        scientific_rows,
        args.output_dir / "heatmap_model_disagreement_elevation.png",
        title="Model disagreement heatmap: elevation",
        metric_getter=lambda row: float_or_none(row.get("elevation_deg")),
        distance_fn=lambda a, b: abs(a - b),
        colorbar_label="Mean absolute difference (deg)",
    )
    plot_material_model_heatmap(
        plt,
        scientific_rows,
        args.output_dir / "heatmap_material_model_travel_time.png",
        metric_key="travel_time_ms_pred",
        title="Material × model heatmap: travel time",
        colorbar_label="Travel time (ms)",
    )
    plot_material_model_heatmap(
        plt,
        scientific_rows,
        args.output_dir / "heatmap_material_model_displacement.png",
        metric_key="max_displacement",
        title="Material × model heatmap: log10(max displacement)",
        colorbar_label="log10(max displacement)",
        log_scale=True,
    )
    plot_material_model_heatmap(
        plt,
        scientific_rows,
        args.output_dir / "heatmap_material_model_temperature.png",
        metric_key="max_temperature_perturbation",
        title="Material × model heatmap: log10(max temperature perturbation)",
        colorbar_label="log10(max temperature perturbation)",
        log_scale=True,
    )
    parameter_heatmaps = {
        "heatmap_time_model_travel_time.png": plot_parameter_model_heatmap(
            plt,
            scientific_rows,
            args.output_dir / "heatmap_time_model_travel_time.png",
            parameter_key="time_ms",
            metric_key="travel_time_ms_pred",
            title="Time × model heatmap: travel time",
            colorbar_label="Travel time (ms)",
        ),
        "heatmap_probe_z_model_travel_time.png": plot_parameter_model_heatmap(
            plt,
            scientific_rows,
            args.output_dir / "heatmap_probe_z_model_travel_time.png",
            parameter_key="probe_z",
            metric_key="travel_time_ms_pred",
            title="Probe z × model heatmap: travel time",
            colorbar_label="Travel time (ms)",
        ),
        "heatmap_temperature_model_temperature_perturbation.png": plot_parameter_model_heatmap(
            plt,
            scientific_rows,
            args.output_dir / "heatmap_temperature_model_temperature_perturbation.png",
            parameter_key="temperature_c",
            metric_key="max_temperature_perturbation",
            title="Temperature × model heatmap: temperature perturbation",
            colorbar_label="Temperature perturbation",
        ),
        "heatmap_pressure_model_displacement.png": plot_parameter_model_heatmap(
            plt,
            scientific_rows,
            args.output_dir / "heatmap_pressure_model_displacement.png",
            parameter_key="pressure_mpa",
            metric_key="max_displacement",
            title="Pressure × model heatmap: log10(max displacement)",
            colorbar_label="log10(max displacement)",
            log_scale=True,
        ),
    }
    generated_files = sorted(path.name for path in args.output_dir.glob("*.png"))
    print(
        json.dumps(
            {
                "status": "ok",
                "input": str(args.input),
                "output_dir": str(args.output_dir),
                "chart_count": len(generated_files),
                "include_fallback": include_fallback,
                "warnings": analysis.warnings,
                "skipped_parameter_heatmaps": [
                    name for name, generated in parameter_heatmaps.items() if not generated
                ],
                "generated_files": generated_files,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
