#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_INPUT = Path("artifacts/data_experiments/inputs/model_comparison_inputs_2d_4materials_balanced.jsonl")
DEFAULT_RESULTS = Path("artifacts/data_experiments/results_2d_4materials_balanced")
DEFAULT_FIGURES = Path("figures/results_2d_4materials_balanced")
DEFAULT_TABLES = Path("tables/results_2d_4materials_balanced")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the strict 2D model-comparison experiment and regenerate 2D charts/tables."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--physics-min-materials", type=int, default=4)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["pinn", "mgn", "fno", "transformer"],
        choices=["pinn", "mgn", "fno", "transformer"],
    )
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-experiment", action="store_true")
    parser.add_argument("--skip-charts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = sys.executable

    if not args.skip_experiment:
        run(
            [
                python,
                "scripts/run_model_service_experiment.py",
                "--input",
                str(args.input),
                "--output-dir",
                str(args.results_dir),
                "--timeout-seconds",
                str(args.timeout_seconds),
                "--models",
                *args.models,
            ]
            + (["--skip-preflight"] if args.skip_preflight else [])
        )

    if not args.skip_charts:
        run(
            [
                python,
                "scripts/generate_2d_result_graphs.py",
                "--summary",
                str(args.results_dir / "summary.csv"),
                "--out-figures",
                str(args.figures_dir),
                "--out-tables",
                str(args.tables_dir),
                "--physics-min-materials",
                str(args.physics_min_materials),
            ]
        )


def run(command: list[str]) -> None:
    print("$", " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
