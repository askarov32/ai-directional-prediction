from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


PRIMARY_OUTPUT_CHANNELS = ("temperature_k", "disp_x", "disp_y", "disp_z")


@dataclass(frozen=True)
class FNOChannelConfig:
    target_channels: tuple[str, ...] = PRIMARY_OUTPUT_CHANNELS
    include_static: bool = True
    include_masks: bool = True
    include_coords: bool = True
    include_time: bool = True


@dataclass(frozen=True)
class ChannelStatistics:
    channel_names: list[str]
    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray
    units: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_names": self.channel_names,
            "mean": [float(value) for value in self.mean],
            "std": [float(value) for value in self.std],
            "min": [float(value) for value in self.min],
            "max": [float(value) for value in self.max],
            "units": self.units,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ChannelStatistics | None":
        if not isinstance(payload, dict):
            return None
        channel_names = payload.get("channel_names")
        mean = payload.get("mean")
        std = payload.get("std")
        minimum = payload.get("min")
        maximum = payload.get("max")
        if not all(isinstance(value, list) for value in (channel_names, mean, std, minimum, maximum)):
            return None
        return cls(
            channel_names=[str(value) for value in channel_names],
            mean=np.asarray(mean, dtype=np.float32),
            std=np.asarray(std, dtype=np.float32),
            min=np.asarray(minimum, dtype=np.float32),
            max=np.asarray(maximum, dtype=np.float32),
            units={str(key): str(value) for key, value in dict(payload.get("units", {})).items()},
        )


def build_fno_sample(tensors, *, time_index: int, channel_config: FNOChannelConfig):
    from fno_service.data.dataset import FNOSample

    dynamic_t = tensors.grid_dynamic[time_index]
    target = _select_dynamic_channels(
        tensors.grid_dynamic[time_index + 1],
        field_names=tensors.field_names,
        selected=channel_config.target_channels,
    )
    input_chunks = [dynamic_t]
    if channel_config.include_static:
        input_chunks.append(tensors.grid_static)
    if channel_config.include_masks:
        input_chunks.append(tensors.grid_masks)
    if channel_config.include_coords:
        input_chunks.append(tensors.grid_coords)
    if channel_config.include_time:
        input_chunks.append(_time_channel(time_index, tensors.grid_dynamic.shape[0], dynamic_t.shape[1:]))

    return FNOSample(
        inputs=np.concatenate(input_chunks, axis=0).astype(np.float32, copy=False),
        target=target.astype(np.float32, copy=False),
        time_index=time_index,
        next_time_index=time_index + 1,
    )


def normalize_channels(values: np.ndarray, stats: ChannelStatistics | None) -> np.ndarray:
    if stats is None:
        return values.astype(np.float32, copy=False)
    mean, std = _broadcast_stats(values, stats)
    return ((values - mean) / np.maximum(std, 1e-8)).astype(np.float32, copy=False)


def denormalize_channels(values: np.ndarray, stats: ChannelStatistics | None) -> np.ndarray:
    if stats is None:
        return values.astype(np.float32, copy=False)
    mean, std = _broadcast_stats(values, stats)
    return (values * np.maximum(std, 1e-8) + mean).astype(np.float32, copy=False)


def _broadcast_stats(values: np.ndarray, stats: ChannelStatistics) -> tuple[np.ndarray, np.ndarray]:
    channel_count = len(stats.channel_names)
    if values.ndim >= 2 and values.shape[0] != channel_count and values.shape[1] == channel_count:
        shape = (1, channel_count) + (1,) * (values.ndim - 2)
    else:
        shape = (channel_count,) + (1,) * (values.ndim - 1)
    return stats.mean.reshape(shape), stats.std.reshape(shape)


def infer_channel_units(channel_names: list[str]) -> dict[str, str]:
    units: dict[str, str] = {}
    for name in channel_names:
        if name == "temperature_k":
            units[name] = "K"
        elif name.startswith("disp_"):
            units[name] = "m"
        elif name.startswith("vel_"):
            units[name] = "m/s"
        elif name in {"youngs_modulus", "thermal_density"}:
            units[name] = "Pa"
        elif name == "poissons_ratio":
            units[name] = "1"
        elif name == "density":
            units[name] = "kg/m^3"
        elif name == "thermal_expansion":
            units[name] = "1/K"
        elif name == "thermal_conductivity":
            units[name] = "W/(m*K)"
        elif name == "heat_capacity":
            units[name] = "J/(kg*K)"
        elif name.startswith("coord_") or name == "time_fraction" or name.endswith("_mask"):
            units[name] = "normalized"
        else:
            units[name] = "unknown"
    return units


def _select_dynamic_channels(values: np.ndarray, *, field_names: list[str], selected: tuple[str, ...]) -> np.ndarray:
    indexes = []
    missing = []
    for name in selected:
        if name in field_names:
            indexes.append(field_names.index(name))
        else:
            missing.append(name)
    if missing:
        raise ValueError(f"Missing target channels in FNO dynamic fields: {missing}")
    return values[indexes]


def _time_channel(time_index: int, total_steps: int, spatial_shape: tuple[int, int, int]) -> np.ndarray:
    denominator = max(total_steps - 1, 1)
    value = np.float32(time_index / denominator)
    return np.full((1, *spatial_shape), value, dtype=np.float32)
