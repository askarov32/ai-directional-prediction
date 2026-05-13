from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pipeline import run_pipeline
from src.training.train import load_config


def parse_args():
    p = argparse.ArgumentParser(description="Prepare graph dataset from real COMSOL CSV + .mphtxt")
    p.add_argument("--dataset_id", required=True)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--registry_dir", default=None)
    p.add_argument("--k_nearest", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    if args.registry_dir:
        data_cfg["registry_dir"] = args.registry_dir
    if args.k_nearest is not None:
        data_cfg["k_nearest"] = args.k_nearest
    meta = run_pipeline(args.dataset_id, data_cfg, data_cfg.get("registry_dir", "datasets"))
    print("✅ Dataset prepared")
    for k in ["dataset_id", "n_nodes", "n_edges", "n_timesteps", "n_dynamic_fields", "n_train", "n_val", "n_test", "node_in_dim"]:
        print(f"{k}: {meta.get(k)}")


if __name__ == "__main__":
    main()
