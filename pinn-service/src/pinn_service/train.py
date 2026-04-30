from __future__ import annotations

import argparse
from pathlib import Path

from pinn_service.trainer import train_pinn
from pinn_service.training_config import TrainingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the first hybrid PINN baseline on prepared COMSOL data.")
    parser.add_argument("--dataset", required=True, help="Path to training_samples.npz")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoint and metrics")
    parser.add_argument("--device", default="cpu", help="Torch device, for example cpu or cuda")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--activation", choices=("tanh", "silu", "gelu", "relu"), default="tanh")
    parser.add_argument("--supervised-weight", type=float, default=1.0)
    parser.add_argument("--velocity-weight", type=float, default=0.25)
    parser.add_argument("--thermal-residual-weight", type=float, default=0.05)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainingConfig(
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        activation=args.activation,
        supervised_weight=args.supervised_weight,
        velocity_weight=args.velocity_weight,
        thermal_residual_weight=args.thermal_residual_weight,
        sample_limit=args.sample_limit,
        seed=args.seed,
    )
    artifacts = train_pinn(config)
    print("Checkpoint:", artifacts.checkpoint_path)
    print("Best checkpoint:", artifacts.best_checkpoint_path)
    print("Metrics:", artifacts.metrics_path)
    print("Config:", artifacts.config_path)
    print("Scalers:", artifacts.scalers_path)


if __name__ == "__main__":
    main()
