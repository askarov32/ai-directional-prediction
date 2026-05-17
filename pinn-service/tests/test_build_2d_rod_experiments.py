from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_2d_rod_experiments.py"
SPEC = importlib.util.spec_from_file_location("build_2d_rod_experiments", SCRIPT_PATH)
assert SPEC is not None
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def test_select_plane_indices_prefers_plane_with_most_nodes() -> None:
    z_values = np.asarray([0.0, 0.5, 0.5, -0.5, 0.5], dtype=np.float32)

    selected = builder.select_plane_indices(
        z_values,
        policy="max_nodes",
        plane_z=0.0,
        decimals=8,
        tolerance=None,
    )

    assert selected.plane_z == 0.5
    assert selected.indices.tolist() == [1, 2, 4]


def test_apply_strict_2d_zeroing_removes_out_of_plane_values() -> None:
    payload = {
        "initial_coordinates": np.ones((2, 3), dtype=np.float32),
        "dynamic_coordinates": np.ones((2, 3, 3), dtype=np.float32),
        "displacement": np.ones((2, 3, 3), dtype=np.float32),
        "velocity": np.ones((2, 3, 3), dtype=np.float32),
        "stress_normal": np.ones((2, 3, 4), dtype=np.float32),
        "stress_shear": np.ones((2, 3, 3), dtype=np.float32),
        "strain": np.ones((2, 3, 6), dtype=np.float32),
    }

    builder.apply_strict_2d_zeroing(payload)

    assert np.allclose(payload["initial_coordinates"][:, 2], 0.0)
    assert np.allclose(payload["dynamic_coordinates"][:, :, 2], 0.0)
    assert np.allclose(payload["displacement"][:, :, 2], 0.0)
    assert np.allclose(payload["velocity"][:, :, 2], 0.0)
    assert np.allclose(payload["stress_normal"][:, :, 3], 0.0)
    assert np.allclose(payload["stress_shear"][:, :, 1:3], 0.0)
    assert np.allclose(payload["strain"][:, :, 2], 0.0)
    assert np.allclose(payload["strain"][:, :, 4:6], 0.0)
