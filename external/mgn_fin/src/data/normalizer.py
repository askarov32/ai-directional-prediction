"""Normalization helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch


class FeatureNormalizer:
    def __init__(self, stats: Dict[str, Dict[str, float]] | None = None, min_std: float = 1e-8):
        self.stats = stats or {}
        self.min_std = float(min_std)

    def fit(self, names: List[str], values: np.ndarray) -> "FeatureNormalizer":
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 2:
            # [N, F]
            axes = (0,)
        elif arr.ndim == 3:
            # [T, N, F]
            axes = (0, 1)
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape={arr.shape}")
        for i, name in enumerate(names):
            v = arr[..., i]
            mean = float(np.nanmean(v))
            std = float(np.nanstd(v))
            if not np.isfinite(std) or std < self.min_std:
                std = 1.0
            if not np.isfinite(mean):
                mean = 0.0
            self.stats[name] = {"mean": mean, "std": std}
        return self

    def normalize_array(self, names: List[str], values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32).copy()
        for i, name in enumerate(names):
            st = self.stats.get(name, {"mean": 0.0, "std": 1.0})
            arr[..., i] = (arr[..., i] - st["mean"]) / st["std"]
        return arr

    def denormalize_array(self, names: List[str], values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32).copy()
        for i, name in enumerate(names):
            st = self.stats.get(name, {"mean": 0.0, "std": 1.0})
            arr[..., i] = arr[..., i] * st["std"] + st["mean"]
        return arr

    def normalize_tensor(self, names: List[str], values: torch.Tensor) -> torch.Tensor:
        out = values.clone()
        for i, name in enumerate(names):
            st = self.stats.get(name, {"mean": 0.0, "std": 1.0})
            out[..., i] = (out[..., i] - st["mean"]) / st["std"]
        return out

    def denormalize_tensor(self, names: List[str], values: torch.Tensor) -> torch.Tensor:
        out = values.clone()
        for i, name in enumerate(names):
            st = self.stats.get(name, {"mean": 0.0, "std": 1.0})
            out[..., i] = out[..., i] * st["std"] + st["mean"]
        return out

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return self.stats

    @classmethod
    def from_dict(cls, stats: Dict[str, Dict[str, float]], min_std: float = 1e-8) -> "FeatureNormalizer":
        return cls(stats=stats, min_std=min_std)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureNormalizer":
        with Path(path).open("r", encoding="utf-8") as f:
            stats = json.load(f)
        return cls(stats=stats)
