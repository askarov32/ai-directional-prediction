from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from pinn_service.train import load_loss_scales_from_report
from pinn_service.trainer import train_pinn
from pinn_service.training_config import TrainingConfig


DEFAULT_TRAIN_DATASET = "pinn-service/artifacts/rod_experiments/splits/train_samples.npz"
DEFAULT_VAL_DATASET = "pinn-service/artifacts/rod_experiments/splits/val_samples.npz"
DEFAULT_LOSS_SCALE_REPORT = "pinn-service/artifacts/rod_experiments/reports/loss_scale_report.json"
DEFAULT_OUTPUT_DIR = "pinn-service/artifacts/checkpoints/rod_all_rocks_experiment"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a full PINN training experiment with train/val split, loss scaling, "
            "checkpoint writing, and post-train report generation."
        )
    )
    parser.add_argument("--train-dataset", default=DEFAULT_TRAIN_DATASET)
    parser.add_argument("--val-dataset", default=DEFAULT_VAL_DATASET)
    parser.add_argument("--loss-scale-report", default=DEFAULT_LOSS_SCALE_REPORT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--validation-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--activation", choices=("tanh", "silu", "gelu", "relu"), default="tanh")
    parser.add_argument("--supervised-weight", type=float, default=1.0)
    parser.add_argument("--velocity-weight", type=float, default=0.25)
    parser.add_argument("--wave-residual-weight", type=float, default=0.1)
    parser.add_argument("--thermal-residual-weight", type=float, default=0.05)
    parser.add_argument("--reference-temperature-k", type=float, default=293.15)
    parser.add_argument("--loss-balance-mode", choices=("fixed", "normalize"), default="normalize")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr-scheduler-patience", type=int, default=40)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--early-stopping-patience", type=int, default=250)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--validation-sample-limit", type=int, default=None)
    parser.add_argument("--progress-interval-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Train only and skip HTML/SVG report generation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print the resolved training config without running training.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_dataset = resolve_existing_path(args.train_dataset, "train dataset")
    val_dataset = resolve_existing_path(args.val_dataset, "validation dataset")
    loss_scale_report = resolve_existing_path(args.loss_scale_report, "loss scale report")
    output_dir = Path(args.output_dir).expanduser().resolve()
    device = resolve_device(args.device)
    loss_scales = load_loss_scales_from_report(str(loss_scale_report))

    config = TrainingConfig(
        dataset_path=train_dataset,
        val_dataset_path=val_dataset,
        output_dir=output_dir,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_batch_size=args.validation_batch_size,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        activation=args.activation,
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
        physics_mode="coupled_thermoelastic",
        sample_limit=args.sample_limit,
        validation_sample_limit=args.validation_sample_limit,
        progress_interval_batches=args.progress_interval_batches,
        seed=args.seed,
    )

    print_experiment_summary(config=config, loss_scale_report=loss_scale_report)
    if args.dry_run:
        return

    artifacts = train_pinn(config)
    print("\nTraining artifacts")
    print("Checkpoint:", artifacts.checkpoint_path)
    print("Best checkpoint:", artifacts.best_checkpoint_path)
    print("Metrics JSON:", artifacts.metrics_path)
    print("Metrics CSV:", artifacts.metrics_csv_path)
    print("Config:", artifacts.config_path)
    print("Scalers:", artifacts.scalers_path)

    if not args.skip_report:
        generate_training_report(artifacts.metrics_path)


def resolve_existing_path(raw_path: str, label: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def resolve_device(raw_device: str) -> str:
    if raw_device != "auto":
        if raw_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device '{raw_device}', but this Python environment does not have CUDA available."
            )
        return raw_device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def print_experiment_summary(*, config: TrainingConfig, loss_scale_report: Path) -> None:
    payload = config.to_dict()
    payload["loss_scale_report"] = str(loss_scale_report)
    print("Resolved PINN training experiment:")
    print(json.dumps(payload, indent=2))


def generate_training_report(metrics_path: Path) -> None:
    script_path = Path(__file__).with_name("generate_training_report.py")
    command = [
        sys.executable,
        str(script_path),
        "--metrics-json",
        str(metrics_path),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
