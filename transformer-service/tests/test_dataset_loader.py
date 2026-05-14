from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import pytest

from transformer_service.dataset import (
    INPUT_CHANNEL_NAMES,
    TARGET_CHANNEL_NAMES,
    AutoregressivePairsDataset,
    build_sandstone_tensors,
    build_train_val_split,
    load_pairs_bundle,
    save_sandstone_artifacts,
)


def _write_csv(
    path: Path,
    coords: list[tuple[float, float, float]],
    field_names: list[str],
    times: list[float],
    values: np.ndarray,
) -> None:
    n_nodes = len(coords)
    assert values.shape == (n_nodes, len(times), len(field_names))
    header_metadata = [
        "% Model,test.mph",
        "% Version,COMSOL test",
        "% Date,test",
        "% Dimension,3",
        f"% Nodes,{n_nodes}",
        f"% Expressions,{len(times)*len(field_names)}",
        "% Description,test",
        "% Length unit,m",
    ]
    columns = ["% X", "Y", "Z"]
    for time in times:
        time_str = f"{time:g}"
        for field in field_names:
            columns.append(f"{field} @ t={time_str}")
    lines = header_metadata + [",".join(columns)]
    for i, (x, y, z) in enumerate(coords):
        cells = [repr(x), repr(y), repr(z)]
        for t_idx in range(len(times)):
            for f_idx in range(len(field_names)):
                cells.append(repr(float(values[i, t_idx, f_idx])))
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_synthetic_sandstone(tmp_path: Path, drop_node: int = 1) -> Path:
    coords_full = [
        (-0.5, -0.5, -0.4),
        (-0.5, -0.5, 0.4),
        (0.5, -0.5, -0.4),
        (0.5, 0.5, 0.4),
        (0.0, 0.0, 0.0),
    ]
    times = [0.0, 1e-4, 2e-4]
    materials_fields = [
        "solid.E (Pa)",
        "solid.nu (1)",
        "solid.rho (kg/m^3)",
        "te1.alpha_iso (1/K)",
    ]
    temperature_fields = [
        "T (K)",
        "x (m)",
        "y (m)",
        "z (m)",
        "ht.k_iso (W/(m*K))",
        "ht.rho (kg/m^3)",
        "ht.Cp (J/(kg*K))",
    ]
    displacement_fields = [
        "u (m)",
        "v (m)",
        "w (m)",
        "ut (m/s)",
        "vt (m/s)",
        "wt (m/s)",
    ]

    rng = np.random.default_rng(0)
    materials_values = rng.normal(size=(len(coords_full), len(times), len(materials_fields)))
    materials_values[..., 0] = 1.5e10
    materials_values[..., 1] = 0.22
    materials_values[..., 2] = 2200.0
    materials_values[..., 3] = 1.0e-5
    temperature_values = rng.normal(size=(len(coords_full), len(times), len(temperature_fields)))
    temperature_values[..., 0] = 293.15 + rng.normal(scale=0.5, size=(len(coords_full), len(times)))
    temperature_values[..., 4] = 2.2
    temperature_values[..., 6] = 800.0
    coords_disp = [c for idx, c in enumerate(coords_full) if idx != drop_node]
    displacement_values = rng.normal(scale=1e-6, size=(len(coords_disp), len(times), len(displacement_fields)))

    sandstone_dir = tmp_path / "synthetic"
    sandstone_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(sandstone_dir / "data_materials.csv", coords_full, materials_fields, times, materials_values)
    _write_csv(sandstone_dir / "data_temperature.csv", coords_full, temperature_fields, times, temperature_values)
    _write_csv(sandstone_dir / "data_displacement.csv", coords_disp, displacement_fields, times, displacement_values)
    return sandstone_dir


def test_build_sandstone_tensors_intersects_correctly(tmp_path):
    sandstone_dir = _make_synthetic_sandstone(tmp_path, drop_node=2)
    tensors = build_sandstone_tensors(sandstone_dir)
    assert tensors.state.shape[0] == 4  # 5 full coords − 1 dropped from displacement
    assert tensors.state.shape[1] == 3  # 3 timesteps
    assert tensors.state.shape[2] == len(INPUT_CHANNEL_NAMES)
    assert tensors.raw_node_counts == {
        "materials": 5,
        "temperature": 5,
        "displacement": 4,
        "intersection": 4,
    }
    # Coordinates must come from the intersection (excluded node not present)
    excluded = np.array([0.5, -0.5, -0.4], dtype=np.float32)
    for row in tensors.coords:
        assert not np.allclose(row, excluded), "Excluded node leaked into intersection"


def test_save_and_load_sandstone_artifacts(tmp_path):
    sandstone_dir = _make_synthetic_sandstone(tmp_path, drop_node=1)
    tensors = build_sandstone_tensors(sandstone_dir)
    out = tmp_path / "artifacts"
    save_sandstone_artifacts(tensors, out)
    bundle = load_pairs_bundle(out / "pairs.npz")
    assert bundle.state.shape == tensors.state.shape
    assert bundle.input_channel_names == INPUT_CHANNEL_NAMES
    assert bundle.target_channel_names == TARGET_CHANNEL_NAMES
    assert np.allclose(bundle.input_mean, tensors.input_mean)


def test_autoregressive_dataset_returns_correct_shapes(tmp_path):
    sandstone_dir = _make_synthetic_sandstone(tmp_path, drop_node=0)
    tensors = build_sandstone_tensors(sandstone_dir)
    out = tmp_path / "artifacts"
    save_sandstone_artifacts(tensors, out)
    bundle = load_pairs_bundle(out / "pairs.npz")
    train, val = build_train_val_split(bundle, train_fraction=0.5)
    assert len(train) >= 1 and len(val) >= 1
    dataset = AutoregressivePairsDataset(bundle, train)
    sample = dataset[0]
    n_nodes = bundle.state.shape[0]
    assert sample["input_tokens"].shape == (n_nodes, len(INPUT_CHANNEL_NAMES))
    assert sample["query_coords"].shape == (n_nodes, 3)
    assert sample["target"].shape == (n_nodes, len(TARGET_CHANNEL_NAMES))
    assert torch.isfinite(sample["input_tokens"]).all()
    assert torch.isfinite(sample["target"]).all()


def test_build_train_val_split_minimum():
    class FakeBundle:
        state = np.zeros((1, 4, 1))
        input_channel_names = INPUT_CHANNEL_NAMES
        target_channel_names = TARGET_CHANNEL_NAMES
        coords = np.zeros((1, 3))
        input_mean = np.zeros(16)
        input_std = np.ones(16)
        target_mean = np.zeros(4)
        target_std = np.ones(4)
        times = np.array([0.0, 1.0, 2.0, 3.0])

    train, val = build_train_val_split(FakeBundle(), train_fraction=0.99)
    assert len(train) >= 1
    assert len(val) >= 1
