from __future__ import annotations

import numpy as np

from pinn_service.training_data import load_training_data


def _write_training_npz(path) -> None:
    input_feature_names = np.array(
        [
            "x",
            "y",
            "z",
            "t",
            "youngs_modulus",
            "poissons_ratio",
            "density",
            "thermal_expansion",
            "thermal_conductivity",
            "heat_capacity",
        ]
    )
    target_feature_names = np.array(
        [
            "temperature_k",
            "disp_x",
            "disp_y",
            "disp_z",
            "vel_x",
            "vel_y",
            "vel_z",
        ]
    )
    inputs = np.arange(100, dtype=np.float32).reshape(10, 10)
    targets = np.arange(70, dtype=np.float32).reshape(10, 7)
    np.savez(
        path,
        inputs=inputs,
        targets=targets,
        input_feature_names=input_feature_names,
        target_feature_names=target_feature_names,
    )


def test_sample_limit_uses_configurable_seed(tmp_path):
    dataset_path = tmp_path / "training_samples.npz"
    _write_training_npz(dataset_path)

    first = load_training_data(dataset_path, sample_limit=4, seed=7)
    second = load_training_data(dataset_path, sample_limit=4, seed=7)
    different = load_training_data(dataset_path, sample_limit=4, seed=8)

    assert np.array_equal(first.inputs, second.inputs)
    assert not np.array_equal(first.inputs, different.inputs)
