from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from fno_service.training import FNOTrainingConfig, train_fno


def main() -> None:
    args = parse_args()
    config_payload = _load_yaml(args.config)
    overrides = {
        "dataset_path": args.dataset_path,
        "output_dir": args.output_dir,
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "width": args.width,
        "modes_x": args.modes_x,
        "modes_y": args.modes_y,
        "depth": args.depth,
        "val_fraction": args.val_fraction,
        "sample_limit": args.sample_limit,
        "seed": args.seed,
    }
    merged = {**config_payload, **{key: value for key, value in overrides.items() if value is not None}}
    config = FNOTrainingConfig(**merged)

    print("Resolved FNO training config:")
    print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
    if args.dry_run:
        return

    artifacts = train_fno(config)
    print("FNO training artifacts:")
    print(json.dumps({key: str(value) for key, value in artifacts.__dict__.items()}, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MVP FNO2d checkpoint on regular grid tensors.")
    parser.add_argument("--config", default="fno-service/configs/train_fno.yaml")
    parser.add_argument("--dataset-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--width", type=int)
    parser.add_argument("--modes-x", type=int)
    parser.add_argument("--modes-y", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--val-fraction", type=float)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"FNO training config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"FNO training config must be a mapping: {config_path}")
    return payload


if __name__ == "__main__":
    main()
