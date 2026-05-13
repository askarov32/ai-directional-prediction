from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.evaluate import evaluate_model
from src.training.train import load_config


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate checkpoint on processed datasets")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset_ids", nargs="*", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    metrics = evaluate_model(cfg, checkpoint=args.checkpoint, dataset_ids=args.dataset_ids)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("✅ Evaluation complete")


if __name__ == "__main__":
    main()
