from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fno_service.data.dataset import FNOTimeStepDataset, load_fno_grid_tensors
from fno_service.data.pinn_to_grid import convert_pinn_structured_to_fno_grid


def write_tiny_pinn_structured(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    structured_path = root / "structured_dataset.npz"
    metadata_path = root / "dataset_metadata.json"

    coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    times = np.asarray([0.0, 0.1, 0.2], dtype=np.float32)
    temperature = np.arange(12, dtype=np.float32).reshape(4, 3)
    displacement = np.ones((4, 3, 3), dtype=np.float32)
    material_static = np.ones((4, 4), dtype=np.float32)
    thermal_properties = np.ones((4, 3, 3), dtype=np.float32) * 2.0

    np.savez_compressed(
        structured_path,
        times=times,
        initial_coordinates=coords,
        dynamic_coordinates=np.repeat(coords[:, None, :], 3, axis=1),
        material_static=material_static,
        temperature=temperature,
        thermal_properties=thermal_properties,
        displacement=displacement,
        velocity=np.zeros((4, 3, 3), dtype=np.float32),
        stress_normal=np.zeros((4, 3, 4), dtype=np.float32),
        stress_shear=np.zeros((4, 3, 3), dtype=np.float32),
        strain=np.zeros((4, 3, 6), dtype=np.float32),
    )
    metadata_path.write_text(json.dumps({"rock_id": "tiny"}), encoding="utf-8")
    return structured_path, metadata_path


def test_convert_pinn_structured_to_fno_grid(tmp_path: Path) -> None:
    structured_path, metadata_path = write_tiny_pinn_structured(tmp_path / "pinn")
    output_dir = tmp_path / "fno"

    metadata = convert_pinn_structured_to_fno_grid(
        structured_path=structured_path,
        metadata_path=metadata_path,
        output_dir=output_dir,
        grid_resolution=(1, 2, 2),
        max_timesteps=2,
    )
    tensors = load_fno_grid_tensors(output_dir)
    dataset = FNOTimeStepDataset(tensors)

    assert metadata["source_format"] == "pinn_structured_dataset"
    assert tensors.grid_dynamic.shape == (2, 4, 1, 2, 2)
    assert tensors.grid_static.shape == (7, 1, 2, 2)
    assert tensors.grid_masks.shape == (1, 1, 2, 2)
    assert dataset[0].target.shape == (4, 1, 2, 2)
