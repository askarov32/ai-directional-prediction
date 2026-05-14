from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fno_service.data.preprocessing import FNOChannelConfig, build_fno_sample


REQUIRED_FNO_FILES = (
    "grid_dynamic.npy",
    "grid_static.npy",
    "grid_masks.npy",
    "grid_coords.npy",
    "field_names.json",
    "static_feature_names.json",
    "mask_names.json",
    "metadata.json",
)


@dataclass(frozen=True)
class FNOGridTensors:
    grid_dynamic: np.ndarray
    grid_static: np.ndarray
    grid_masks: np.ndarray
    grid_coords: np.ndarray
    field_names: list[str]
    static_feature_names: list[str]
    mask_names: list[str]
    metadata: dict
    selected_time_indices: np.ndarray | None = None
    source_node_index: np.ndarray | None = None


@dataclass(frozen=True)
class FNOSample:
    inputs: np.ndarray
    target: np.ndarray
    time_index: int
    next_time_index: int


class FNOTimeStepDataset:
    """Time-step dataset over regular FNO grids.

    Each sample uses dynamic fields at time t plus static channels, masks,
    coordinates, and a time channel to predict primary fields at t+1.
    """

    def __init__(
        self,
        tensors: FNOGridTensors,
        *,
        channel_config: FNOChannelConfig | None = None,
        time_indices: list[int] | None = None,
    ) -> None:
        self.tensors = tensors
        self.channel_config = channel_config or FNOChannelConfig()
        max_start = tensors.grid_dynamic.shape[0] - 1
        if max_start < 1:
            raise ValueError("FNOTimeStepDataset requires at least two timesteps.")
        self.time_indices = time_indices or list(range(max_start))
        invalid = [index for index in self.time_indices if index < 0 or index >= max_start]
        if invalid:
            raise ValueError(f"Invalid FNO time indices: {invalid[:5]}")

    @classmethod
    def from_directory(
        cls,
        dataset_dir: str | Path,
        *,
        channel_config: FNOChannelConfig | None = None,
        time_indices: list[int] | None = None,
    ) -> "FNOTimeStepDataset":
        return cls(load_fno_grid_tensors(dataset_dir), channel_config=channel_config, time_indices=time_indices)

    def __len__(self) -> int:
        return len(self.time_indices)

    def __getitem__(self, index: int) -> FNOSample:
        time_index = self.time_indices[index]
        return build_fno_sample(
            tensors=self.tensors,
            time_index=time_index,
            channel_config=self.channel_config,
        )


def load_fno_grid_tensors(dataset_dir: str | Path) -> FNOGridTensors:
    root = Path(dataset_dir).expanduser().resolve()
    missing = [name for name in REQUIRED_FNO_FILES if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"FNO dataset is missing required files in {root}: {missing}")

    tensors = FNOGridTensors(
        grid_dynamic=np.load(root / "grid_dynamic.npy").astype(np.float32, copy=False),
        grid_static=np.load(root / "grid_static.npy").astype(np.float32, copy=False),
        grid_masks=np.load(root / "grid_masks.npy").astype(np.float32, copy=False),
        grid_coords=np.load(root / "grid_coords.npy").astype(np.float32, copy=False),
        field_names=_read_json_list(root / "field_names.json"),
        static_feature_names=_read_json_list(root / "static_feature_names.json"),
        mask_names=_read_json_list(root / "mask_names.json"),
        metadata=json.loads((root / "metadata.json").read_text(encoding="utf-8")),
        selected_time_indices=_load_optional_array(root / "selected_time_indices.npy"),
        source_node_index=_load_optional_array(root / "source_node_index.npy"),
    )
    validate_fno_grid_tensors(tensors)
    return tensors


def validate_fno_grid_tensors(tensors: FNOGridTensors) -> None:
    dynamic = tensors.grid_dynamic
    static = tensors.grid_static
    masks = tensors.grid_masks
    coords = tensors.grid_coords

    if dynamic.ndim != 5:
        raise ValueError("grid_dynamic must have shape [T,C,Z,Y,X].")
    if static.ndim != 4:
        raise ValueError("grid_static must have shape [S,Z,Y,X].")
    if masks.ndim != 4:
        raise ValueError("grid_masks must have shape [M,Z,Y,X].")
    if coords.ndim != 4 or coords.shape[0] != 3:
        raise ValueError("grid_coords must have shape [3,Z,Y,X].")
    spatial_shape = dynamic.shape[2:]
    if static.shape[1:] != spatial_shape or masks.shape[1:] != spatial_shape or coords.shape[1:] != spatial_shape:
        raise ValueError("FNO grid tensors must share the same [Z,Y,X] shape.")
    if dynamic.shape[1] != len(tensors.field_names):
        raise ValueError("field_names length must match grid_dynamic channel count.")
    if static.shape[0] != len(tensors.static_feature_names):
        raise ValueError("static_feature_names length must match grid_static channel count.")
    if masks.shape[0] != len(tensors.mask_names):
        raise ValueError("mask_names length must match grid_masks channel count.")
    for name, array in {
        "grid_dynamic": dynamic,
        "grid_static": static,
        "grid_masks": masks,
        "grid_coords": coords,
    }.items():
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or infinite values.")


def _read_json_list(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise ValueError(f"Expected a JSON string list in {path}")
    return payload


def _load_optional_array(path: Path) -> np.ndarray | None:
    return np.load(path) if path.exists() else None
