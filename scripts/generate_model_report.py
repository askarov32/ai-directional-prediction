#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from analyze_model_outputs import load_and_analyze_summary, model_sort_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a markdown report and figures for model comparison outputs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/data_experiments/results/summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
    )
    parser.add_argument(
        "--include-fallback",
        type=str,
        default="false",
        choices=["true", "false"],
    )
    parser.add_argument(
        "--save-svg",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Reserved for future vector export. PNG export remains the active output mode.",
    )
    parser.add_argument(
        "--save-png",
        type=str,
        default="true",
        choices=["true", "false"],
    )
    return parser.parse_args()


def run_chart_generator(input_path: Path, figures_dir: Path, include_fallback: bool) -> dict:
    command = [
        sys.executable,
        "scripts/generate_model_comparison_charts.py",
        "--input",
        str(input_path),
        "--output-dir",
        str(figures_dir),
        "--include-fallback",
        "true" if include_fallback else "false",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def build_response_table(analysis) -> list[str]:
    lines = [
        "| Model | Total | Checkpoint | Fallback | Error | Timeout | Outlier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    by_model = analysis.stats["by_model"]
    for model in sorted(by_model.keys(), key=model_sort_key):
        row_stats = by_model[model]
        outlier_count = sum(1 for row in analysis.rows if row["model"] == model and row["is_outlier"])
        lines.append(
            f"| `{model}` | {row_stats.get('total', 0)} | {row_stats.get('ok_checkpoint', 0)} | "
            f"{row_stats.get('ok_fallback', 0)} | {row_stats.get('error', 0)} | "
            f"{row_stats.get('timeout', 0)} | {outlier_count} |"
        )
    return lines


def build_outlier_list(analysis) -> list[str]:
    outliers = analysis.stats.get("outlier_cases", [])
    if not outliers:
        return ["- Outlier cases were not detected under current sanity limits."]
    lines = []
    for item in outliers[:30]:
        reasons = ", ".join(item["reasons"])
        lines.append(
            f"- `{item['case_id']}` / `{item['model']}` / `{item['material']}`: "
            f"reasons = `{reasons}`, max_displacement = `{item['max_displacement']}`, "
            f"max_temperature_perturbation = `{item['max_temperature_perturbation']}`"
        )
    if len(outliers) > 30:
        lines.append(f"- ... and {len(outliers) - 30} more outlier rows.")
    return lines


def build_key_observations(analysis, chart_result: dict, include_fallback: bool) -> list[str]:
    observations = []
    if include_fallback:
        observations.append("- Fallback responses were included in scientific plots for this run.")
    else:
        observations.append("- Fallback responses were excluded from scientific plots by default.")

    if any("fno elevation is always zero" in warning.lower() for warning in analysis.warnings):
        observations.append(
            "- FNO elevation is always zero in the analyzed run, which strongly suggests 2D adaptation rather than true 3D directional prediction."
        )
    if any("fno has displacement outliers" in warning.lower() for warning in analysis.warnings):
        observations.append(
            "- FNO produces displacement outliers and should not be interpreted as physically calibrated until scaling is verified."
        )
    if any("fno has temperature perturbation outliers" in warning.lower() for warning in analysis.warnings):
        observations.append(
            "- FNO produces temperature perturbation outliers, so raw cross-model scale comparisons remain diagnostic only."
        )
    if chart_result.get("skipped_parameter_heatmaps"):
        observations.append(
            "- Some parameter sensitivity heatmaps were skipped because the current experiment pack did not contain enough unique parameter values."
        )
    observations.append(
        "- Circular statistics are used for azimuth disagreement, so wrap-around effects near ±180 degrees are handled correctly."
    )
    return observations


def build_limitations(analysis) -> list[str]:
    limitations = [
        "- Any fallback model must be treated as diagnostic only and not as a scientific comparator.",
        "- Any outlier values should not be interpreted as physically valid until scaling and normalization are verified.",
        "- These plots summarize service outputs, not ground-truth error against laboratory or COMSOL reference targets.",
    ]
    if any(row.get("domain_adaptation") == "rect_3d_to_rect_2d" for row in analysis.rows if row["model"] == "fno"):
        limitations.append(
            "- FNO is currently operating with `rect_3d_to_rect_2d` adaptation on these runs, so it is not a full 3D predictor yet."
        )
    return limitations


def build_figures_section(figures_dir: Path) -> list[str]:
    lines = []
    for image_path in sorted(figures_dir.glob("*.png")):
        lines.append(f"## {image_path.name}")
        lines.append(f"![{image_path.name}](figures/{image_path.name})")
        lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    if args.save_png != "true":
        raise SystemExit("PNG export must remain enabled for the current report generator.")

    output_dir = args.output_dir
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    include_fallback = args.include_fallback == "true"
    chart_result = run_chart_generator(args.input, figures_dir, include_fallback)
    analysis = load_and_analyze_summary(args.input)

    report_lines = [
        "# Model Comparison Report",
        "",
        "## Dataset summary",
        f"- Input file: `{args.input}`",
        f"- Case count: `{analysis.stats['case_count']}`",
        f"- Response count: `{analysis.stats['response_count']}`",
        f"- Include fallback in scientific plots: `{include_fallback}`",
        "",
        "## Response summary by model",
        *build_response_table(analysis),
        "",
        "## Warnings",
    ]
    if analysis.warnings:
        report_lines.extend(f"- {warning}" for warning in analysis.warnings)
    else:
        report_lines.append("- No warnings were generated.")
    report_lines.extend(
        [
            "",
            "## Outlier cases",
            *build_outlier_list(analysis),
            "",
            "## Key observations",
            *build_key_observations(analysis, chart_result, include_fallback),
            "",
            "## Limitations",
            *build_limitations(analysis),
            "",
            "## Chart generation details",
            f"- Generated figures: `{chart_result.get('chart_count', 0)}`",
            f"- Skipped parameter heatmaps: `{', '.join(chart_result.get('skipped_parameter_heatmaps', [])) or 'none'}`",
            "",
            "## Generated plots",
            "",
        ]
    )
    report_lines.extend(build_figures_section(figures_dir))

    report_path = output_dir / "model_comparison_report.md"
    report_path.write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "report": str(report_path),
                "figures_dir": str(figures_dir),
                "chart_count": chart_result.get("chart_count", 0),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
