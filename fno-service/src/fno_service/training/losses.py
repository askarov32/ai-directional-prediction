from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def field_mse_loss(prediction: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """Mean squared field loss with an optional spatial validity mask."""
    squared_error = (prediction - target).pow(2)
    if mask is None:
        return squared_error.mean()
    broadcast_mask = mask.to(dtype=squared_error.dtype, device=squared_error.device)
    while broadcast_mask.ndim < squared_error.ndim:
        broadcast_mask = broadcast_mask.unsqueeze(1)
    denominator = torch.clamp(broadcast_mask.sum() * prediction.shape[1], min=1.0)
    return (squared_error * broadcast_mask).sum() / denominator


def relative_l2_loss(prediction: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    numerator = torch.linalg.vector_norm((prediction - target).reshape(prediction.shape[0], -1), dim=1)
    denominator = torch.linalg.vector_norm(target.reshape(target.shape[0], -1), dim=1).clamp_min(eps)
    return (numerator / denominator).mean()
