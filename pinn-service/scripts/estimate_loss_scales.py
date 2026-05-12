from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import torch

from pinn_service.losses import compute_hybrid_pinn_loss
from pinn_service.model import MLP_PINN
from pinn_service.training_data import PRIMARY_OUTPUT_NAMES, load_training_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate initial PINN loss component scales on sampled training data.")
    parser.add_argument(
        "--dataset",
        default="pinn-service/artifacts/rod_experiments/splits/train_samples.npz",
        help="Training samples dataset. Use the train split when available.",
    )
    parser.add_argument(
        "--output-dir",
        default="pinn-service/artifacts/rod_experiments/reports",
        help="Directory where loss_scale_report.json/html are written.",
    )
    parser.add_argument("--sample-limit", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--activation", choices=("tanh", "silu", "gelu", "relu"), default="tanh")
    parser.add_argument("--supervised-weight", type=float, default=1.0)
    parser.add_argument("--velocity-weight", type=float, default=0.25)
    parser.add_argument("--wave-residual-weight", type=float, default=0.1)
    parser.add_argument("--thermal-residual-weight", type=float, default=0.05)
    parser.add_argument("--reference-temperature-k", type=float, default=293.15)
    parser.add_argument("--physics-mode", choices=("coupled_thermoelastic", "simple_heat"), default="coupled_thermoelastic")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = estimate_loss_scales(args)
    json_path = output_dir / "loss_scale_report.json"
    html_path = output_dir / "loss_scale_report.html"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")

    print("Loss scale JSON:", json_path)
    print("Loss scale HTML:", html_path)


def estimate_loss_scales(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    data = load_training_data(args.dataset, sample_limit=args.sample_limit, seed=args.seed)
    loader = data.make_loader(batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = MLP_PINN(
        input_dim=len(data.input_feature_names),
        output_dim=len(PRIMARY_OUTPUT_NAMES),
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        activation=args.activation,
    ).to(device)

    input_mean = torch.tensor(data.input_scaler.mean, dtype=torch.float32, device=device)
    input_std = torch.tensor(data.input_scaler.std, dtype=torch.float32, device=device)
    output_mean = torch.tensor(data.output_scaler.mean, dtype=torch.float32, device=device)
    output_std = torch.tensor(data.output_scaler.std, dtype=torch.float32, device=device)

    aggregates = {
        "supervised_loss": 0.0,
        "velocity_consistency_loss": 0.0,
        "wave_residual_loss": 0.0,
        "thermal_residual_loss": 0.0,
        "total_loss": 0.0,
    }
    batch_count = 0

    for inputs_scaled, primary_targets_scaled, velocity_targets in loader:
        if batch_count >= args.max_batches:
            break
        loss, metrics = compute_hybrid_pinn_loss(
            model=model,
            inputs_scaled=inputs_scaled.to(device),
            primary_targets_scaled=primary_targets_scaled.to(device),
            velocity_targets=velocity_targets.to(device),
            input_scaler_mean=input_mean,
            input_scaler_std=input_std,
            output_scaler_mean=output_mean,
            output_scaler_std=output_std,
            supervised_weight=args.supervised_weight,
            velocity_weight=args.velocity_weight,
            wave_residual_weight=args.wave_residual_weight,
            thermal_residual_weight=args.thermal_residual_weight,
            reference_temperature_k=args.reference_temperature_k,
            physics_mode=args.physics_mode,
        )
        del loss
        for key, value in metrics.items():
            aggregates[key] += value
        batch_count += 1

    averages = {key: value / max(batch_count, 1) for key, value in aggregates.items()}
    recommended_weights = recommend_weights(averages)
    return {
        "dataset": str(Path(args.dataset).expanduser().resolve()),
        "sample_limit": args.sample_limit,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "actual_batches": batch_count,
        "seed": args.seed,
        "device": args.device,
        "physics_mode": args.physics_mode,
        "loss_averages": averages,
        "current_weights": {
            "supervised_weight": args.supervised_weight,
            "velocity_weight": args.velocity_weight,
            "wave_residual_weight": args.wave_residual_weight,
            "thermal_residual_weight": args.thermal_residual_weight,
        },
        "rough_balancing_weights": recommended_weights,
        "note": "rough_balancing_weights scale each residual near the supervised loss magnitude at random initialization; treat them as diagnostics, not final hyperparameters.",
    }


def recommend_weights(averages: dict[str, float]) -> dict[str, float]:
    supervised = max(averages.get("supervised_loss", 0.0), 1e-12)
    return {
        "supervised_weight": 1.0,
        "velocity_weight": supervised / max(averages.get("velocity_consistency_loss", 0.0), 1e-12),
        "wave_residual_weight": supervised / max(averages.get("wave_residual_loss", 0.0), 1e-12),
        "thermal_residual_weight": supervised / max(averages.get("thermal_residual_loss", 0.0), 1e-12),
    }


def render_html(report: dict) -> str:
    loss_rows = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{value:.6g}</td></tr>"
        for key, value in report["loss_averages"].items()
    )
    weight_rows = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{value:.6g}</td></tr>"
        for key, value in report["rough_balancing_weights"].items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PINN Initial Loss Scale Report</title>
  <style>
    body {{ margin: 0; background: #0f172a; color: #e5eefb; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(900px, calc(100% - 48px)); margin: 40px auto 72px; }}
    .card {{ background: #172033; border: 1px solid #334155; border-radius: 18px; padding: 22px; margin-top: 18px; }}
    p, td {{ color: #cbd5e1; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    td, th {{ border-bottom: 1px solid #263244; padding: 11px 12px; text-align: right; }}
    td:first-child, th:first-child {{ text-align: left; }}
    th {{ color: #bfdbfe; }}
    code {{ color: #dbeafe; }}
  </style>
</head>
<body>
<main>
  <h1>PINN Initial Loss Scale Report</h1>
  <p>Dataset: <code>{html.escape(report["dataset"])}</code></p>
  <section class="card">
    <h2>Average Loss Components</h2>
    <table><thead><tr><th>component</th><th>value</th></tr></thead><tbody>{loss_rows}</tbody></table>
  </section>
  <section class="card">
    <h2>Rough Balancing Weights</h2>
    <p>{html.escape(report["note"])}</p>
    <table><thead><tr><th>weight</th><th>value</th></tr></thead><tbody>{weight_rows}</tbody></table>
  </section>
</main>
</body>
</html>
"""


if __name__ == "__main__":
    main()
