from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PRIMARY_OUTPUT_CHANNELS = ("temperature_k", "disp_x", "disp_y", "disp_z")


@dataclass(frozen=True)
class FNOChannelConfig:
    target_channels: tuple[str, ...] = PRIMARY_OUTPUT_CHANNELS
    include_static: bool = True
    include_masks: bool = True
    include_coords: bool = True
    include_time: bool = True


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
