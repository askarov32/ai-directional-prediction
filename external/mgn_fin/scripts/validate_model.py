from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train import load_config
from src.validation import run_full_validation


def parse_args():
    p = argparse.ArgumentParser(description="Full COMSOL-vs-MeshGraphNet validation")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--dataset_id", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--output_dir", default="outputs/validation")
    p.add_argument("--max_rollout_steps", type=int, default=None, help="Limit rollout length for fast validation")
    p.add_argument("--slice_axis", default="z", choices=["x", "y", "z"])
    p.add_argument("--slice_value", type=float, default=None)
    p.add_argument("--slice_tol", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    summary = run_full_validation(
        config=cfg,
        dataset_id=args.dataset_id,
        checkpoint=args.checkpoint,
        split=args.split,
        max_rollout_steps=args.max_rollout_steps,
        output_dir=args.output_dir,
        slice_axis=args.slice_axis,
        slice_value=args.slice_value,
        slice_tol=args.slice_tol,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("✅ Full validation complete")
    print(f"Report: {Path(args.output_dir) / 'validation_report.html'}")


if __name__ == "__main__":
    main()
