"""Losses and metrics."""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def infer_group(field_name: str) -> str:
    f = field_name.lower()
    if f in {"t", "temp", "temperature"} or "temperature" in f:
        return "temperature"
    if f in {"u", "v", "w"}:
        return "displacement"
    if f in {"ut", "vt", "wt"} or "velocity" in f:
        return "velocity"
    if f.startswith("s") or "stress" in f:
        return "stress"
    if f.startswith("e") or "strain" in f:
        return "strain"
    return "other"


class WeightedFieldMSE(nn.Module):
    def __init__(self, field_names: List[str], weights: Dict[str, float] | None = None):
        super().__init__()
        self.field_names = field_names
        self.weights = weights or {}
        self.groups = [infer_group(f) for f in field_names]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
        denom = 0.0
        for i, group in enumerate(self.groups):
            w = float(self.weights.get(group, self.weights.get("other", 1.0)))
            loss = loss + w * F.mse_loss(pred[:, i], target[:, i])
            denom += w
        return loss / max(denom, 1e-8)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, field_names: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with torch.no_grad():
        err = pred - target
        out["mse"] = float(torch.mean(err ** 2).detach().cpu())
        out["rmse"] = float(torch.sqrt(torch.mean(err ** 2)).detach().cpu())
        out["mae"] = float(torch.mean(torch.abs(err)).detach().cpu())
        for i, f in enumerate(field_names):
            e = err[:, i]
            out[f"rmse/{f}"] = float(torch.sqrt(torch.mean(e ** 2)).detach().cpu())
            out[f"mae/{f}"] = float(torch.mean(torch.abs(e)).detach().cpu())
    return out
