from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from transformer_service.dataset import INPUT_CHANNEL_NAMES, TARGET_CHANNEL_NAMES


@dataclass(frozen=True)
class NormalizationStats:
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray


def normalize_state(state: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    if state.shape[-1] != stats.input_mean.shape[-1]:
        raise ValueError(
            f"State channel count {state.shape[-1]} != input_mean channels {stats.input_mean.shape[-1]}"
        )
    return ((state - stats.input_mean) / stats.input_std).astype(np.float32)


def denormalize_target(target_norm: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    if target_norm.shape[-1] != stats.target_mean.shape[-1]:
        raise ValueError(
            f"Target channel count {target_norm.shape[-1]} != target_mean channels {stats.target_mean.shape[-1]}"
        )
    return (target_norm * stats.target_std + stats.target_mean).astype(np.float32)


def update_state_with_prediction(
    state_norm: np.ndarray,
    prediction_raw: np.ndarray,
    stats: NormalizationStats,
) -> np.ndarray:
    """Take normalized input state (N, F_in) and raw next-step target (N, F_out=T,u,v,w),
    fold the prediction back into a normalized state for the next autoregressive step.
    Material parameters stay frozen; coordinates and material channels are preserved.
    """
    state_raw = state_norm * stats.input_std + stats.input_mean
    target_indices = [INPUT_CHANNEL_NAMES.index(name) for name in TARGET_CHANNEL_NAMES]
    state_raw[:, target_indices] = prediction_raw
    return normalize_state(state_raw, stats)
