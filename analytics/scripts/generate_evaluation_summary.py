from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


SENSITIVITY_SWEEPS = ("temperature", "pressure", "frequency")
METRICS = ("azimuth_deg", "travel_time_ms", "magnitude", "max_displacement", "max_temperature_perturbation")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate evaluation summary reports from model comparison metrics.")
    parser.add_argument("--metrics", default="analytics/outputs/model_comparison_metrics.csv")
    parser.add_argument("--predictions", default="analytics/outputs/model_comparison_predictions.json")
    parser.add_argument("--output-json", default="analytics/outputs/evaluation_summary.json")
    parser.add_argument("--output-html", default="analytics/outputs/evaluation_summary.html")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics_path = Path(args.metrics).expanduser().resolve()
    predictions_path = Path(args.predictions).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_html = Path(args.output_html).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    summary = build_summary(metrics_path=metrics_path, predictions_path=predictions_path)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_html.write_text(render_html(summary), encoding="utf-8")

    print("Evaluation summary JSON:", output_json)
    print("Evaluation summary HTML:", output_html)


def build_summary(*, metrics_path: Path, predictions_path: Path) -> dict[str, Any]:
    rows = load_rows(metrics_path)
    predictions_payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    overview = {
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "result_count": len(predictions_payload.get("results", [])),
        "ok_rows": len(rows),
        "models": sorted({row["model"] for row in rows}),
        "media": sorted({row["medium_id"] for row in rows}),
        "sweeps": sorted({row["sweep_name"] for row in rows}),
    }

    baseline_rows = [row for row in rows if row["sweep_name"] == "baseline"]
    baseline_by_medium = build_baseline_by_medium(baseline_rows)
    model_aggregates = build_model_aggregates(rows)
    medium_aggregates = build_medium_aggregates(rows)
    sensitivity = build_sensitivity_summary(rows)
    highlights = build_highlights(model_aggregates, sensitivity)

    return {
        "overview": overview,
        "highlights": highlights,
        "baseline_by_medium": baseline_by_medium,
        "model_aggregates": model_aggregates,
        "medium_aggregates": medium_aggregates,
        "sensitivity": sensitivity,
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            if row.get("status") != "ok":
                continue
            parsed = dict(row)
            for field in (
                "temperature_c",
                "pressure_mpa",
                "frequency_hz",
                "azimuth_deg",
                "elevation_deg",
                "magnitude",
                "travel_time_ms",
                "direction_x",
                "direction_y",
                "direction_z",
                "max_displacement",
                "max_temperature_perturbation",
                "latency_ms_backend",
                "latency_ms_client",
            ):
                parsed[field] = to_float(parsed.get(field))
            rows.append(parsed)
    return rows


def build_baseline_by_medium(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = []
    for medium_id in sorted({row["medium_id"] for row in rows}):
        entries = []
        medium_rows = [row for row in rows if row["medium_id"] == medium_id]
        for model in sorted({row["model"] for row in medium_rows}):
            row = next(item for item in medium_rows if item["model"] == model)
            entries.append(
                {
                    "model": model,
                    "azimuth_deg": row["azimuth_deg"],
                    "travel_time_ms": row["travel_time_ms"],
                    "magnitude": row["magnitude"],
                    "max_displacement": row["max_displacement"],
                    "max_temperature_perturbation": row["max_temperature_perturbation"],
                    "latency_ms_backend": row["latency_ms_backend"],
                }
            )
        ordered.append({"medium_id": medium_id, "entries": entries})
    return ordered


def build_model_aggregates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)

    aggregates = []
    for model in sorted(grouped):
        items = grouped[model]
        aggregates.append(
            {
                "model": model,
                "scenario_count": len(items),
                "avg_azimuth_deg": avg(items, "azimuth_deg"),
                "avg_travel_time_ms": avg(items, "travel_time_ms"),
                "avg_magnitude": avg(items, "magnitude"),
                "avg_max_displacement": avg(items, "max_displacement"),
                "avg_max_temperature_perturbation": avg(items, "max_temperature_perturbation"),
                "avg_latency_ms_backend": avg(items, "latency_ms_backend"),
                "avg_latency_ms_client": avg(items, "latency_ms_client"),
                "max_travel_time_ms": max_value(items, "travel_time_ms"),
                "min_travel_time_ms": min_value(items, "travel_time_ms"),
            }
        )
    return aggregates


def build_medium_aggregates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["medium_id"]].append(row)

    aggregates = []
    for medium_id in sorted(grouped):
        items = grouped[medium_id]
        aggregates.append(
            {
                "medium_id": medium_id,
                "scenario_count": len(items),
                "avg_azimuth_deg": avg(items, "azimuth_deg"),
                "avg_travel_time_ms": avg(items, "travel_time_ms"),
                "avg_magnitude": avg(items, "magnitude"),
                "avg_max_displacement": avg(items, "max_displacement"),
                "avg_max_temperature_perturbation": avg(items, "max_temperature_perturbation"),
                "avg_latency_ms_backend": avg(items, "latency_ms_backend"),
            }
        )
    return aggregates


def build_sensitivity_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_sweep: dict[str, list[dict[str, Any]]] = {}
    for sweep_name in SENSITIVITY_SWEEPS:
        entries = []
        sweep_rows = [row for row in rows if row["sweep_name"] == sweep_name]
        for model in sorted({row["model"] for row in sweep_rows}):
            model_rows = [row for row in sweep_rows if row["model"] == model]
            medium_entries = []
            spans_by_metric: dict[str, list[float]] = defaultdict(list)
            for medium_id in sorted({row["medium_id"] for row in model_rows}):
                medium_rows = [row for row in model_rows if row["medium_id"] == medium_id]
                metric_spans = {metric: span(medium_rows, metric) for metric in METRICS}
                for metric, value in metric_spans.items():
                    spans_by_metric[metric].append(value)
                medium_entries.append({"medium_id": medium_id, "metric_spans": metric_spans})

            average_spans = {metric: round(mean(values), 6) if values else 0.0 for metric, values in spans_by_metric.items()}
            score = (
                average_spans["travel_time_ms"]
                + average_spans["azimuth_deg"] / 100.0
                + average_spans["max_temperature_perturbation"]
                + average_spans["max_displacement"] * 1000.0
            )
            entries.append(
                {
                    "model": model,
                    "average_spans": average_spans,
                    "score": round(score, 6),
                    "per_medium": medium_entries,
                }
            )
        entries.sort(key=lambda item: item["score"], reverse=True)
        per_sweep[sweep_name] = entries

    ranking = {
        sweep_name: [entry["model"] for entry in entries]
        for sweep_name, entries in per_sweep.items()
    }
    return {"per_sweep": per_sweep, "ranking": ranking}


def build_highlights(model_aggregates: list[dict[str, Any]], sensitivity: dict[str, Any]) -> dict[str, Any]:
    fastest = min(model_aggregates, key=lambda item: item["avg_latency_ms_backend"])
    slowest = max(model_aggregates, key=lambda item: item["avg_latency_ms_backend"])
    hottest = max(model_aggregates, key=lambda item: item["avg_max_temperature_perturbation"])
    travel_leader = min(model_aggregates, key=lambda item: item["avg_travel_time_ms"])

    return {
        "fastest_backend_model": {
            "model": fastest["model"],
            "avg_latency_ms_backend": fastest["avg_latency_ms_backend"],
        },
        "slowest_backend_model": {
            "model": slowest["model"],
            "avg_latency_ms_backend": slowest["avg_latency_ms_backend"],
        },
        "largest_average_temperature_response": {
            "model": hottest["model"],
            "avg_max_temperature_perturbation": hottest["avg_max_temperature_perturbation"],
        },
        "shortest_average_travel_time": {
            "model": travel_leader["model"],
            "avg_travel_time_ms": travel_leader["avg_travel_time_ms"],
        },
        "sensitivity_leaders": {
            sweep_name: entries[0]["model"] if entries else None
            for sweep_name, entries in sensitivity["per_sweep"].items()
        },
    }


def render_html(summary: dict[str, Any]) -> str:
    baseline_sections = "".join(render_baseline_section(item) for item in summary["baseline_by_medium"])
    model_rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['model'])}</td>"
        f"<td>{item['scenario_count']}</td>"
        f"<td>{fmt(item['avg_travel_time_ms'])}</td>"
        f"<td>{fmt(item['avg_azimuth_deg'])}</td>"
        f"<td>{fmt(item['avg_magnitude'])}</td>"
        f"<td>{fmt(item['avg_max_displacement'])}</td>"
        f"<td>{fmt(item['avg_max_temperature_perturbation'])}</td>"
        f"<td>{fmt(item['avg_latency_ms_backend'])}</td>"
        "</tr>"
        for item in summary["model_aggregates"]
    )
    sensitivity_sections = "".join(render_sensitivity_section(name, items) for name, items in summary["sensitivity"]["per_sweep"].items())
    highlights = summary["highlights"]
    highlight_cards = [
        ("Fastest backend model", f"{highlights['fastest_backend_model']['model']} ({fmt(highlights['fastest_backend_model']['avg_latency_ms_backend'])} ms)"),
        ("Slowest backend model", f"{highlights['slowest_backend_model']['model']} ({fmt(highlights['slowest_backend_model']['avg_latency_ms_backend'])} ms)"),
        ("Shortest average travel time", f"{highlights['shortest_average_travel_time']['model']} ({fmt(highlights['shortest_average_travel_time']['avg_travel_time_ms'])} ms)"),
        ("Largest average temperature response", f"{highlights['largest_average_temperature_response']['model']} ({fmt(highlights['largest_average_temperature_response']['avg_max_temperature_perturbation'])})"),
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model Evaluation Summary</title>
  <style>
    body {{ margin: 0; background: #0f172a; color: #e5eefb; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1180px, calc(100% - 48px)); margin: 40px auto 72px; }}
    h1, h2, h3 {{ margin: 0; }}
    h1 {{ font-size: 34px; letter-spacing: -0.03em; }}
    h2 {{ margin-top: 28px; font-size: 24px; }}
    h3 {{ margin-top: 20px; color: #bfdbfe; }}
    .card {{ background: #172033; border: 1px solid #334155; border-radius: 18px; padding: 22px; margin-top: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
    .metric {{ background: #111827; border: 1px solid #263244; border-radius: 14px; padding: 14px; }}
    .metric span {{ display: block; color: #94a3b8; font-size: 13px; }}
    .metric strong {{ display: block; margin-top: 6px; font-size: 18px; color: #f8fafc; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #263244; text-align: right; color: #cbd5e1; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: #bfdbfe; }}
    p {{ color: #94a3b8; }}
  </style>
</head>
<body>
<main>
  <h1>Model Evaluation Summary</h1>
  <p>Rows analyzed: {summary['overview']['ok_rows']} successful predictions across models {", ".join(summary['overview']['models'])}.</p>
  <section class="card">
    <h2>Highlights</h2>
    <div class="grid">
      {''.join(f'<div class="metric"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>' for label, value in highlight_cards)}
    </div>
  </section>
  <section class="card">
    <h2>Model Aggregates</h2>
    <table>
      <thead><tr><th>model</th><th>scenarios</th><th>avg travel time</th><th>avg azimuth</th><th>avg magnitude</th><th>avg displacement</th><th>avg temp perturbation</th><th>avg backend latency</th></tr></thead>
      <tbody>{model_rows}</tbody>
    </table>
  </section>
  <section class="card">
    <h2>Baseline By Medium</h2>
    {baseline_sections}
  </section>
  <section class="card">
    <h2>Sensitivity Ranking</h2>
    <p>Higher score means the model changes more across the sweep according to average spans in travel time, azimuth, displacement, and temperature perturbation.</p>
    {sensitivity_sections}
  </section>
</main>
</body>
</html>
"""


def render_baseline_section(item: dict[str, Any]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(entry['model'])}</td>"
        f"<td>{fmt(entry['travel_time_ms'])}</td>"
        f"<td>{fmt(entry['azimuth_deg'])}</td>"
        f"<td>{fmt(entry['magnitude'])}</td>"
        f"<td>{fmt(entry['max_displacement'])}</td>"
        f"<td>{fmt(entry['max_temperature_perturbation'])}</td>"
        f"<td>{fmt(entry['latency_ms_backend'])}</td>"
        "</tr>"
        for entry in item["entries"]
    )
    return f"""
    <h3>{html.escape(item['medium_id'])}</h3>
    <table>
      <thead><tr><th>model</th><th>travel time</th><th>azimuth</th><th>magnitude</th><th>displacement</th><th>temp perturbation</th><th>backend latency</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """


def render_sensitivity_section(sweep_name: str, items: list[dict[str, Any]]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{index}</td>"
        f"<td>{html.escape(item['model'])}</td>"
        f"<td>{fmt(item['score'])}</td>"
        f"<td>{fmt(item['average_spans']['travel_time_ms'])}</td>"
        f"<td>{fmt(item['average_spans']['azimuth_deg'])}</td>"
        f"<td>{fmt(item['average_spans']['magnitude'])}</td>"
        f"<td>{fmt(item['average_spans']['max_displacement'])}</td>"
        f"<td>{fmt(item['average_spans']['max_temperature_perturbation'])}</td>"
        "</tr>"
        for index, item in enumerate(items, start=1)
    )
    return f"""
    <h3>{html.escape(sweep_name.title())}</h3>
    <table>
      <thead><tr><th>rank</th><th>model</th><th>score</th><th>travel span</th><th>azimuth span</th><th>magnitude span</th><th>displacement span</th><th>temp span</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """


def avg(rows: list[dict[str, Any]], field: str) -> float:
    return round(mean(row[field] for row in rows if row[field] is not None), 6)


def min_value(rows: list[dict[str, Any]], field: str) -> float:
    return round(min(row[field] for row in rows if row[field] is not None), 6)


def max_value(rows: list[dict[str, Any]], field: str) -> float:
    return round(max(row[field] for row in rows if row[field] is not None), 6)


def span(rows: list[dict[str, Any]], field: str) -> float:
    values = [row[field] for row in rows if row[field] is not None]
    if not values:
        return 0.0
    return round(max(values) - min(values), 6)


def to_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}"


if __name__ == "__main__":
    main()
