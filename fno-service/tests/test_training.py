from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from fno_service.training import FNOTrainingConfig, train_fno
from fno_service.training.checkpoints import load_checkpoint

from .helpers import write_tiny_2d_fno_dataset


def test_train_fno_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    dataset_dir = write_tiny_2d_fno_dataset(tmp_path / "dataset")
    output_dir = tmp_path / "checkpoint"

    artifacts = train_fno(
        FNOTrainingConfig(
            dataset_path=dataset_dir,
            output_dir=output_dir,
            epochs=1,
            batch_size=1,
            width=8,
            modes_x=2,
            modes_y=2,
            depth=1,
            device="cpu",
            seed=7,
        )
    )

    assert artifacts.model_path.exists()
    assert artifacts.best_model_path.exists()
    assert artifacts.metrics_path.exists()
    assert artifacts.metrics_csv_path.exists()
    checkpoint = load_checkpoint(artifacts.best_model_path)
    assert checkpoint["model_config"]["architecture"] == "FNO2d"
    assert checkpoint["channel_metadata"]["target_channels"] == ["temperature_k", "disp_x", "disp_y", "disp_z"]
    normalization = checkpoint["channel_metadata"]["normalization"]
    assert normalization["mode"] == "channel_wise_standardization"
    assert normalization["input"]["channel_names"][0] == "temperature_k"
    assert normalization["target"]["channel_names"] == ["temperature_k", "disp_x", "disp_y", "disp_z"]
    assert normalization["target"]["units"]["temperature_k"] == "K"
    assert normalization["target"]["units"]["disp_x"] == "m"
    metrics = json.loads(artifacts.metrics_path.read_text(encoding="utf-8"))
    assert len(metrics) == 1
    assert np.isfinite(metrics[0]["train_loss"])
    assert all(torch.isfinite(value).all() for value in checkpoint["model_state_dict"].values())
