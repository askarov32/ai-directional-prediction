"""Checkpoint helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(path: str | Path, model, optimizer=None, epoch: int = 0, val_loss: float | None = None, metadata: Dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "val_loss": val_loss,
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model, optimizer=None, map_location="cpu", strict: bool = True) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"], strict=strict)
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt
