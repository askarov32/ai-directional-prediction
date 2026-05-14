from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from fno_service.training import FNOTrainingConfig, train_fno
from fno_service.training.checkpoints import load_checkpoint


def write_tiny_2d_fno_dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    time_steps = 4
    y_size = 6
    x_size = 6
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, y_size, dtype=np.float32),
        np.linspace(0.0, 1.0, x_size, dtype=np.float32),
        indexing="ij",
    )
    dynamic = np.zeros((time_steps, 4, 1, y_size, x_size), dtype=np.float32)
    for time_index in range(time_steps):
        dynamic[time_index, 0, 0] = 293.15 + time_index + xx + yy
        dynamic[time_index, 1, 0] = 0.1 * time_index + xx
        dynamic[time_index, 2, 0] = 0.2 * time_index + yy
        dynamic[time_index, 3, 0] = 0.05 * time_index

    static = np.ones((2, 1, y_size, x_size), dtype=np.float32)
    masks = np.ones((1, 1, y_size, x_size), dtype=np.float32)
    coords = np.stack([xx[None], yy[None], np.zeros((1, y_size, x_size), dtype=np.float32)], axis=0)

    np.save(root / "grid_dynamic.npy", dynamic)
    np.save(root / "grid_static.npy", static)
    np.save(root / "grid_masks.npy", masks)
    np.save(root / "grid_coords.npy", coords)
    (root / "field_names.json").write_text(
        json.dumps(["temperature_k", "disp_x", "disp_y", "disp_z"]),
        encoding="utf-8",
    )
    (root / "static_feature_names.json").write_text(json.dumps(["youngs_modulus", "density"]), encoding="utf-8")
    (root / "mask_names.json").write_text(json.dumps(["grid_valid_mask"]), encoding="utf-8")
    (root / "metadata.json").write_text(json.dumps({"format": "fno_grid", "test": True}), encoding="utf-8")
    return root


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
    metrics = json.loads(artifacts.metrics_path.read_text(encoding="utf-8"))
    assert len(metrics) == 1
    assert np.isfinite(metrics[0]["train_loss"])
    assert all(torch.isfinite(value).all() for value in checkpoint["model_state_dict"].values())
