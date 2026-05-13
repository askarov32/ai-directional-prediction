"""Evaluation utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch

from src.models.losses import compute_metrics
from .checkpoint_manager import load_checkpoint
from .train import build_model, load_training_datasets, setup_device


def evaluate_model(config: Dict, checkpoint: str, dataset_ids: List[str] | None = None) -> Dict:
    data_cfg = config.get("data", {})
    training = config.get("training", {})
    registry_dir = data_cfg.get("registry_dir", "datasets")
    device = setup_device(training.get("device", "auto"))
    bundle = load_training_datasets(dataset_ids, registry_dir)
    model = build_model(config, bundle["node_in_dim"], bundle["edge_in_dim"], bundle["out_dim"]).to(device)
    load_checkpoint(checkpoint, model, map_location=device, strict=True)
    model.eval()

    agg = {}
    count = 0
    with torch.no_grad():
        for split in ["train", "val", "test"]:
            split_metrics = {}
            n = 0
            for s in bundle["splits"][split]:
                x = s["x"].to(device)
                y = s["y"].to(device)
                ei = s["edge_index"].to(device)
                ea = s["edge_attr"].to(device)
                pred = model(x, ei, ea)
                m = compute_metrics(pred, y, bundle["field_names"])
                for k, v in m.items():
                    split_metrics[k] = split_metrics.get(k, 0.0) + v
                n += 1
            if n:
                for k in split_metrics:
                    split_metrics[k] /= n
            agg[split] = split_metrics

    out_dir = Path(config.get("training", {}).get("log_dir", "outputs/logs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "evaluation_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    return agg
