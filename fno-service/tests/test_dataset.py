from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fno_service.data.dataset import FNOTimeStepDataset, load_fno_grid_tensors


def write_tiny_fno_dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "grid_dynamic.npy", np.ones((3, 4, 2, 3, 4), dtype=np.float32))
    np.save(root / "grid_static.npy", np.ones((2, 2, 3, 4), dtype=np.float32))
    np.save(root / "grid_masks.npy", np.ones((1, 2, 3, 4), dtype=np.float32))
    np.save(root / "grid_coords.npy", np.ones((3, 2, 3, 4), dtype=np.float32))
    np.save(root / "selected_time_indices.npy", np.arange(3, dtype=np.int64))
    np.save(root / "source_node_index.npy", np.arange(24, dtype=np.int64).reshape(2, 3, 4))
    (root / "field_names.json").write_text(
        json.dumps(["temperature_k", "disp_x", "disp_y", "disp_z"]),
        encoding="utf-8",
    )
    (root / "static_feature_names.json").write_text(json.dumps(["youngs_modulus", "density"]), encoding="utf-8")
    (root / "mask_names.json").write_text(json.dumps(["grid_valid_mask"]), encoding="utf-8")
    (root / "metadata.json").write_text(json.dumps({"format": "fno_grid"}), encoding="utf-8")
    return root


def test_load_fno_grid_tensors_validates_shapes(tmp_path: Path) -> None:
    dataset_dir = write_tiny_fno_dataset(tmp_path / "fno")

    tensors = load_fno_grid_tensors(dataset_dir)

    assert tensors.grid_dynamic.shape == (3, 4, 2, 3, 4)
    assert tensors.grid_static.shape == (2, 2, 3, 4)
    assert tensors.field_names == ["temperature_k", "disp_x", "disp_y", "disp_z"]


def test_time_step_dataset_builds_model_ready_sample(tmp_path: Path) -> None:
    dataset_dir = write_tiny_fno_dataset(tmp_path / "fno")

    dataset = FNOTimeStepDataset.from_directory(dataset_dir)
    sample = dataset[0]

    assert len(dataset) == 2
    assert sample.inputs.shape == (11, 2, 3, 4)
    assert sample.target.shape == (4, 2, 3, 4)
    assert sample.time_index == 0
    assert sample.next_time_index == 1
