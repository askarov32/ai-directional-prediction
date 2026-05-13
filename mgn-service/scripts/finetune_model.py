from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train import load_config
from src.training.finetune import fine_tune_model


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Conditional MeshGraphNet on new COMSOL datasets")
    p.add_argument("--config", default="configs/finetune.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset_id", default=None, help="Single dataset id, e.g. basalt_comsol_real")
    p.add_argument("--dataset_ids", nargs="*", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--mode", choices=["full", "decoder_only", "processor_decoder"], default=None,
                   help="Fine-tune mode: full | decoder_only | processor_decoder")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    if args.lr is not None:
        cfg.setdefault("training", {})["lr"] = args.lr
    dataset_ids = args.dataset_ids
    if args.dataset_id:
        dataset_ids = [args.dataset_id]
    if args.mode:
        cfg.setdefault("fine_tune", {})["mode"] = args.mode
    result = fine_tune_model(cfg, dataset_ids=dataset_ids, checkpoint=args.checkpoint)
    print("✅ Fine-tuning complete")
    print(result)


if __name__ == "__main__":
    main()
