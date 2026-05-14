from __future__ import annotations

import torch
from torch import Tensor


TARGET_CHANNEL_NAMES = ["temperature_k", "disp_x", "disp_y", "disp_z"]


def supervised_mse(
    pred: Tensor,
    target: Tensor,
    channel_weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch in supervised_mse: pred={tuple(pred.shape)} target={tuple(target.shape)}"
        )

    per_channel = ((pred - target) ** 2).mean(dim=tuple(range(pred.dim() - 1)))
    if channel_weights is not None:
        if channel_weights.shape != per_channel.shape:
            raise ValueError(
                f"channel_weights shape {tuple(channel_weights.shape)} != {tuple(per_channel.shape)}"
            )
        per_channel = per_channel * channel_weights

    total = per_channel.sum()
    metrics = {
        "total_loss": float(total.detach().cpu().item()),
    }
    for index, name in enumerate(TARGET_CHANNEL_NAMES):
        metrics[f"mse_{name}"] = float(per_channel[index].detach().cpu().item())
    return total, metrics
