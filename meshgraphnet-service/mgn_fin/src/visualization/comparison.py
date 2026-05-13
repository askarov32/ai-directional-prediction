"""COMSOL vs AI comparison helpers."""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def compare_trajectories(pred: np.ndarray, truth: np.ndarray, field_names: List[str]) -> Dict[str, float]:
    T = min(pred.shape[0], truth.shape[0])
    N = min(pred.shape[1], truth.shape[1])
    F = min(pred.shape[2], truth.shape[2])
    p = pred[:T, :N, :F]
    y = truth[:T, :N, :F]
    err = p - y
    out = {
        "rmse_total": float(np.sqrt(np.mean(err ** 2))),
        "mae_total": float(np.mean(np.abs(err))),
    }
    for i, name in enumerate(field_names[:F]):
        e = err[:, :, i]
        denom = np.sqrt(np.mean(y[:, :, i] ** 2)) + 1e-12
        out[f"rmse/{name}"] = float(np.sqrt(np.mean(e ** 2)))
        out[f"mae/{name}"] = float(np.mean(np.abs(e)))
        out[f"relative_rmse/{name}"] = float(out[f"rmse/{name}"] / denom)
    return out
