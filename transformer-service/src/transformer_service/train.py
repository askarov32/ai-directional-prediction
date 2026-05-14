from __future__ import annotations

import argparse
from pathlib import Path

from transformer_service.trainer import train_transformer
from transformer_service.training_config import TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Transformer baseline.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to pairs.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--enc-depth", type=int, default=4)
    parser.add_argument("--dec-depth", type=int, default=4)
    parser.add_argument("--ffn-expansion", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-tokens",
        type=int,
        default=1024,
        help="Random token subsample per training step (None / -1 / 0 = use all nodes).",
    )
    args = parser.parse_args()

    n_tokens = args.n_tokens if args.n_tokens and args.n_tokens > 0 else None
    config = TrainingConfig(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        device=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        n_heads=args.n_heads,
        enc_depth=args.enc_depth,
        dec_depth=args.dec_depth,
        ffn_expansion=args.ffn_expansion,
        dropout=args.dropout,
        activation=args.activation,
        grad_clip_norm=args.grad_clip_norm,
        early_stop_patience=args.early_stop_patience,
        seed=args.seed,
        n_tokens=n_tokens,
    )
    artifacts = train_transformer(config)
    print(f"Checkpoint: {artifacts.best_checkpoint_path}")
    print(f"Metrics:    {artifacts.metrics_path}")
    print(f"Config:     {artifacts.config_path}")
    print(f"Scalers:    {artifacts.scalers_path}")


if __name__ == "__main__":
    main()
