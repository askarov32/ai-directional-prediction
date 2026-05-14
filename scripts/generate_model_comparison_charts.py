#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path


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
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def float_or_none(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def ok_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("status") == "ok"]


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


def average_metric(rows: list[dict[str, str]], metric: str, *, by: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        key = row.get(by, "")
        value = float_or_none(row.get(metric, ""))
        if key and value is not None:
            buckets[key].append(value)
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


def grouped_mean(
    rows: list[dict[str, str]],
    *,
    group_keys: tuple[str, ...],
    metric: str,
) -> dict[tuple[str, ...], float]:
    buckets: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = float_or_none(row.get(metric, ""))
        if value is None:
            continue
        key = tuple(row.get(name, "") for name in group_keys)
        if all(key):
            buckets[key].append(value)
    return {key: sum(values) / len(values) for key, values in buckets.items() if values}


def plot_temperature_comparison(plt, rows: list[dict[str, str]], output_path: Path) -> None:
    means = average_metric(rows, "max_temperature_perturbation", by="model")
    if not means:
        render_placeholder(plt, output_path, "Temperature comparison", "No successful temperature metrics were available.")
        return
    labels = list(means.keys())
    values = [means[label] for label in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color="#4F46E5")
    ax.set_title("Temperature perturbation comparison")
    ax.set_ylabel("Mean max temperature perturbation")
    apply_log_scale_if_needed(ax, values)
    annotate_bars(ax, values)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


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
    if not means:
        render_placeholder(plt, output_path, "Displacement magnitude comparison", "No displacement metrics were available.")
        return
    labels = list(means.keys())
    values = [means[label] for label in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color="#0F766E")
    ax.set_title("Max displacement comparison (successful responses only)")
    ax.set_ylabel("Mean max displacement")
    apply_log_scale_if_needed(ax, values)
    annotate_bars(ax, values)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_material_comparison(plt, rows: list[dict[str, str]], output_path: Path) -> None:
    materials = sorted({row["material"] for row in rows if row.get("material")})
    models = sorted({row["model"] for row in rows if row.get("model")})
    if len(materials) < 2:
        render_placeholder(plt, output_path, "Material comparison", "Need at least two materials with successful results.")
        return
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = float_or_none(row.get("travel_time_ms_pred", ""))
        if value is not None:
            buckets[(row["material"], row["model"])].append(value)
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
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            case_ids.append(case_id)
            disagreements.append(math.sqrt(variance))
    if not disagreements:
        render_placeholder(plt, output_path, "Model disagreement", "Need at least two successful model outputs per case.")
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(case_ids, disagreements, color="#B45309")
    ax.set_title("Azimuth disagreement across models")
    ax.set_ylabel("Std. dev. of azimuth (deg)")
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
            grouped[(row["material"], row["model"])].append((time_value, travel_time, max_displacement))
    enough_variation = any(len({time for time, *_ in points}) >= 2 for points in grouped.values())
    if not enough_variation:
        render_placeholder(
            plt,
            output_path,
            "Prediction vs time",
            "Current experiment input pack uses a mostly fixed time point. Generate a multi-time grid to produce this chart.",
        )
        return
    materials = sorted({material for material, _ in grouped})
    fig, axes = plt.subplots(len(materials), 2, figsize=(14, 5 * len(materials)), squeeze=False)
    for row_index, material in enumerate(materials):
        travel_ax = axes[row_index][0]
        disp_ax = axes[row_index][1]
        for _, model in sorted(key for key in grouped if key[0] == material):
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
    models = sorted({key[1] for key in means})
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
    models = sorted({key[1] for key in means})
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
    if not means:
        render_placeholder(plt, output_path, "Elevation comparison", "No elevation metrics were available.")
        return
    labels = list(means.keys())
    values = [means[label] for label in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color="#7C3AED")
    ax.set_title("Mean elevation angle by model")
    ax.set_ylabel("Elevation (deg)")
    annotate_bars(ax, values)
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


def plot_service_status_summary(plt, rows: list[dict[str, str]], output_path: Path) -> None:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        model = row.get("model", "")
        if row.get("status") == "ok" and row.get("fallback_used") == "True":
            status = "fallback"
        else:
            status = row.get("status", "unknown")
        if model:
            counts[model][status] += 1
    if not counts:
        render_placeholder(plt, output_path, "Service status summary", "No rows were available in summary.csv.")
        return
    labels = sorted(counts.keys())
    ok_values = [counts[label].get("ok", 0) for label in labels]
    fallback_values = [counts[label].get("fallback", 0) for label in labels]
    error_values = [counts[label].get("error", 0) for label in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, ok_values, label="ok", color="#15803D")
    ax.bar(labels, fallback_values, bottom=ok_values, label="fallback", color="#B45309")
    error_bottom = [ok + fallback for ok, fallback in zip(ok_values, fallback_values)]
    ax.bar(labels, error_values, bottom=error_bottom, label="error", color="#B91C1C")
    ax.set_title("Service status summary")
    ax.set_ylabel("Case count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    successful_rows = ok_rows(rows)
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
        successful_rows,
        args.output_dir / "temperature_comparison.png",
    )
    plot_direction_components_proxy(
        plt,
        successful_rows,
        args.output_dir / "displacement_components_comparison.png",
    )
    plot_displacement_magnitude(
        plt,
        successful_rows,
        args.output_dir / "displacement_magnitude_comparison.png",
    )
    plot_material_comparison(
        plt,
        successful_rows,
        args.output_dir / "material_comparison_sandstone_vs_basalt.png",
    )
    plot_model_disagreement(
        plt,
        successful_rows,
        args.output_dir / "model_disagreement.png",
    )
    plot_prediction_vs_time(
        plt,
        successful_rows,
        args.output_dir / "prediction_vs_time.png",
    )
    plot_service_status_summary(
        plt,
        rows,
        args.output_dir / "service_status_summary.png",
    )
    plot_basalt_vs_sandstone_travel_time(
        plt,
        successful_rows,
        args.output_dir / "basalt_vs_sandstone_travel_time.png",
    )
    plot_basalt_vs_sandstone_displacement(
        plt,
        successful_rows,
        args.output_dir / "basalt_vs_sandstone_displacement.png",
    )
    plot_elevation_comparison(
        plt,
        successful_rows,
        args.output_dir / "elevation_comparison.png",
    )
    plot_depth_sensitivity(
        plt,
        successful_rows,
        args.output_dir / "depth_sensitivity.png",
    )
    plot_domain_adaptation_summary(
        plt,
        rows,
        args.output_dir / "domain_adaptation_summary.png",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "input": str(args.input),
                "output_dir": str(args.output_dir),
                "chart_count": 12,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
