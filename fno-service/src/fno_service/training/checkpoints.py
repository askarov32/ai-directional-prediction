from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    epoch: int,
    metric: float,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    channel_metadata: dict[str, Any],
    dataset_metadata: dict[str, Any],
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metric": metric,
            "model_config": model_config,
            "training_config": training_config,
            "channel_metadata": channel_metadata,
            "dataset_metadata": dataset_metadata,
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_checkpoint(path: str | Path, *, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def write_json(path: str | Path, payload: dict[str, Any] | list[dict[str, Any]]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path
