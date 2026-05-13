from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train import load_config, train_model


def parse_args():
    p = argparse.ArgumentParser(description="Train base Conditional MeshGraphNet")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--dataset_ids", nargs="*", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    if args.lr is not None:
        cfg.setdefault("training", {})["lr"] = args.lr
    result = train_model(cfg, dataset_ids=args.dataset_ids, fine_tune=False)
    print("✅ Training complete")
    print(result)


if __name__ == "__main__":
    main()
