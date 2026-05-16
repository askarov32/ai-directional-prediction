from __future__ import annotations

import argparse
import json
from pathlib import Path

from pinn_service.model import parse_layer_dims
from pinn_service.trainer import train_pinn
from pinn_service.training_config import TrainingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the first hybrid PINN baseline on prepared COMSOL data.")
    parser.add_argument("--dataset", required=True, help="Path to training_samples.npz")
    parser.add_argument("--val-dataset", default=None, help="Optional validation samples npz used for best checkpoint selection.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoint and metrics")
    parser.add_argument("--device", default="cpu", help="Torch device, for example cpu or cuda")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--validation-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--architecture", choices=("mlp", "res_split"), default="mlp")
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument(
        "--mlp-layer-dims",
        default=None,
        help="Optional comma-separated hidden sizes for the mlp baseline, for example 256,256,192,192,128,128.",
    )
    parser.add_argument("--num-blocks", type=int, default=4, help="Residual trunk block count for --architecture res_split.")
    parser.add_argument("--activation", choices=("tanh", "silu", "gelu", "relu"), default="tanh")
    parser.add_argument("--use-fourier-features", action="store_true")
    parser.add_argument("--fourier-num-frequencies", type=int, default=6)
    parser.add_argument("--fourier-scale", type=float, default=1.0)
    parser.add_argument("--supervised-weight", type=float, default=1.0)
    parser.add_argument("--velocity-weight", type=float, default=0.25)
    parser.add_argument("--wave-residual-weight", type=float, default=0.1)
    parser.add_argument("--thermal-residual-weight", type=float, default=0.05)
    parser.add_argument("--reference-temperature-k", type=float, default=293.15)
    parser.add_argument(
        "--loss-balance-mode",
        choices=("fixed", "normalize"),
        default="fixed",
        help="Use raw component losses or normalize them by precomputed scale factors.",
    )
    parser.add_argument(
        "--loss-scale-report",
        default=None,
        help="Optional loss_scale_report.json used to auto-fill component scales for normalized training.",
    )
    parser.add_argument("--supervised-loss-scale", type=float, default=None)
    parser.add_argument("--velocity-loss-scale", type=float, default=None)
    parser.add_argument("--wave-residual-loss-scale", type=float, default=None)
    parser.add_argument("--thermal-residual-loss-scale", type=float, default=None)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Clip gradients to this norm before optimizer step. Use 0 or a negative value to disable clipping.",
    )
    parser.add_argument("--lr-scheduler-patience", type=int, default=25)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--physics-mode",
        choices=("coupled_thermoelastic", "simple_heat"),
        default="coupled_thermoelastic",
    )
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--validation-sample-limit", type=int, default=None)
    parser.add_argument("--progress-interval-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    loss_scales = resolve_loss_scales(args)
    config = TrainingConfig(
        dataset_path=Path(args.dataset),
        val_dataset_path=Path(args.val_dataset) if args.val_dataset else None,
        output_dir=Path(args.output_dir),
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_batch_size=args.validation_batch_size,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        architecture=args.architecture,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        mlp_layer_dims=parse_layer_dims(args.mlp_layer_dims),
        num_blocks=args.num_blocks,
        activation=args.activation,
        use_fourier_features=args.use_fourier_features,
        fourier_num_frequencies=args.fourier_num_frequencies,
        fourier_scale=args.fourier_scale,
        supervised_weight=args.supervised_weight,
        velocity_weight=args.velocity_weight,
        wave_residual_weight=args.wave_residual_weight,
        thermal_residual_weight=args.thermal_residual_weight,
        reference_temperature_k=args.reference_temperature_k,
        loss_balance_mode=args.loss_balance_mode,
        supervised_loss_scale=loss_scales["supervised_loss_scale"],
        velocity_loss_scale=loss_scales["velocity_loss_scale"],
        wave_residual_loss_scale=loss_scales["wave_residual_loss_scale"],
        thermal_residual_loss_scale=loss_scales["thermal_residual_loss_scale"],
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_patience=args.lr_scheduler_patience,
        lr_scheduler_factor=args.lr_scheduler_factor,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        physics_mode=args.physics_mode,
        sample_limit=args.sample_limit,
        validation_sample_limit=args.validation_sample_limit,
        progress_interval_batches=args.progress_interval_batches,
        seed=args.seed,
    )
    artifacts = train_pinn(config)
    print("Checkpoint:", artifacts.checkpoint_path)
    print("Best checkpoint:", artifacts.best_checkpoint_path)
    print("Metrics:", artifacts.metrics_path)
    print("Metrics CSV:", artifacts.metrics_csv_path)
    print("Config:", artifacts.config_path)
    print("Scalers:", artifacts.scalers_path)


def resolve_loss_scales(args: argparse.Namespace) -> dict[str, float]:
    manual = {
        "supervised_loss_scale": args.supervised_loss_scale,
        "velocity_loss_scale": args.velocity_loss_scale,
        "wave_residual_loss_scale": args.wave_residual_loss_scale,
        "thermal_residual_loss_scale": args.thermal_residual_loss_scale,
    }
    report_values = load_loss_scales_from_report(args.loss_scale_report) if args.loss_scale_report else {}
    resolved = {}
    for key in manual:
        candidate = manual[key]
        if candidate is None:
            candidate = report_values.get(key)
        if candidate is None:
            candidate = 1.0
        resolved[key] = float(candidate)
    return resolved


def load_loss_scales_from_report(path_value: str) -> dict[str, float]:
    path = Path(path_value).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    averages = payload.get("loss_averages", {})
    return {
        "supervised_loss_scale": float(averages.get("supervised_loss", 1.0)),
        "velocity_loss_scale": float(averages.get("velocity_consistency_loss", 1.0)),
        "wave_residual_loss_scale": float(averages.get("wave_residual_loss", 1.0)),
        "thermal_residual_loss_scale": float(averages.get("thermal_residual_loss", 1.0)),
    }


if __name__ == "__main__":
    main()
