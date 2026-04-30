from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


PRIMARY_OUTPUT_NAMES = ["temperature_k", "disp_x", "disp_y", "disp_z"]
VELOCITY_OUTPUT_NAMES = ["vel_x", "vel_y", "vel_z"]


@dataclass(frozen=True)
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "StandardScaler":
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "mean": self.mean.astype(float).tolist(),
            "std": self.std.astype(float).tolist(),
        }


@dataclass(frozen=True)
class LoadedTrainingData:
    inputs: np.ndarray
    inputs_scaled: np.ndarray
    primary_targets: np.ndarray
    primary_targets_scaled: np.ndarray
    velocity_targets: np.ndarray
    input_feature_names: list[str]
    target_feature_names: list[str]
    input_scaler: StandardScaler
    output_scaler: StandardScaler

    def make_loader(self, batch_size: int, shuffle: bool = True) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(self.inputs_scaled),
            torch.from_numpy(self.primary_targets_scaled),
            torch.from_numpy(self.velocity_targets),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def load_training_data(dataset_path: str | Path, sample_limit: int | None = None) -> LoadedTrainingData:
    npz_path = Path(dataset_path).expanduser().resolve()
    payload = np.load(npz_path)

    inputs = payload["inputs"].astype(np.float32)
    targets = payload["targets"].astype(np.float32)
    input_feature_names = payload["input_feature_names"].tolist()
    target_feature_names = payload["target_feature_names"].tolist()

    if sample_limit is not None and sample_limit < len(inputs):
        rng = np.random.default_rng(42)
        indices = rng.choice(len(inputs), size=sample_limit, replace=False)
        inputs = inputs[indices]
        targets = targets[indices]

    primary_indices = [target_feature_names.index(name) for name in PRIMARY_OUTPUT_NAMES]
    velocity_indices = [target_feature_names.index(name) for name in VELOCITY_OUTPUT_NAMES]

    primary_targets = targets[:, primary_indices]
    velocity_targets = targets[:, velocity_indices]

    input_scaler = StandardScaler.fit(inputs)
    output_scaler = StandardScaler.fit(primary_targets)

    return LoadedTrainingData(
        inputs=inputs,
        inputs_scaled=input_scaler.transform(inputs),
        primary_targets=primary_targets,
        primary_targets_scaled=output_scaler.transform(primary_targets),
        velocity_targets=velocity_targets,
        input_feature_names=input_feature_names,
        target_feature_names=target_feature_names,
        input_scaler=input_scaler,
        output_scaler=output_scaler,
    )


def save_scalers(output_dir: str | Path, data: LoadedTrainingData) -> Path:
    path = Path(output_dir).expanduser().resolve() / "scalers.json"
    path.write_text(
        json.dumps(
            {
                "input_feature_names": data.input_feature_names,
                "input_scaler": data.input_scaler.to_dict(),
                "output_feature_names": PRIMARY_OUTPUT_NAMES,
                "output_scaler": data.output_scaler.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
