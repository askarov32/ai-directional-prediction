from __future__ import annotations

import math

import torch
from torch import Tensor


def mae(prediction: Tensor, target: Tensor) -> float:
    return float(torch.mean(torch.abs(prediction - target)).detach().cpu())


def rmse(prediction: Tensor, target: Tensor) -> float:
    return float(torch.sqrt(torch.mean((prediction - target).pow(2))).detach().cpu())


def relative_l2(prediction: Tensor, target: Tensor, eps: float = 1e-8) -> float:
    numerator = torch.linalg.vector_norm((prediction - target).reshape(prediction.shape[0], -1), dim=1)
    denominator = torch.linalg.vector_norm(target.reshape(target.shape[0], -1), dim=1).clamp_min(eps)
    return float((numerator / denominator).mean().detach().cpu())


def finite_metric(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"Metric must be finite, got {value!r}")
    return value
