from __future__ import annotations

import json
from pathlib import Path

import numpy as np


FIELD_NAMES = ["temperature_k", "disp_x", "disp_y", "disp_z"]
STATIC_FEATURE_NAMES = [
    "youngs_modulus",
    "poissons_ratio",
    "density",
    "thermal_expansion",
    "thermal_conductivity",
    "thermal_density",
    "heat_capacity",
]
MASK_NAMES = ["grid_valid_mask"]


def convert_pinn_structured_to_fno_grid(
    *,
    structured_path: str | Path,
    metadata_path: str | Path | None,
    output_dir: str | Path,
    grid_resolution: tuple[int, int, int] = (8, 16, 16),
    max_timesteps: int | None = 64,
) -> dict:
    structured = Path(structured_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    payload = np.load(structured)
    coords = payload["initial_coordinates"].astype(np.float32)
    times = payload["times"].astype(np.float32)
    selected = _select_time_indices(len(times), max_timesteps)
    grid_coords, flat = _regular_grid(coords, grid_resolution)
    nearest_idx, nearest_dist = _nearest_indices(coords, flat)
    valid = _valid_grid_mask(coords, nearest_dist, grid_resolution)

    dynamic_node = np.concatenate(
        [
            payload["temperature"][:, :, None],
            payload["displacement"],
        ],
        axis=2,
    ).astype(np.float32)
    grid_dynamic = dynamic_node[:, selected, :][nearest_idx]
    grid_dynamic = grid_dynamic.reshape(*grid_resolution, len(selected), len(FIELD_NAMES))
    grid_dynamic = np.transpose(grid_dynamic, (3, 4, 0, 1, 2)).astype(np.float32)

    thermal_static = payload["thermal_properties"][:, 0, :].astype(np.float32)
    static_node = np.concatenate([payload["material_static"].astype(np.float32), thermal_static], axis=1)
    grid_static = static_node[nearest_idx].reshape(*grid_resolution, len(STATIC_FEATURE_NAMES))
    grid_static = np.transpose(grid_static, (3, 0, 1, 2)).astype(np.float32)

    grid_masks = valid.reshape(1, *grid_resolution).astype(np.float32)
    source_node_index = nearest_idx.reshape(*grid_resolution).astype(np.int64)

    np.save(output / "grid_dynamic.npy", grid_dynamic)
    np.save(output / "grid_static.npy", grid_static)
    np.save(output / "grid_masks.npy", grid_masks)
    np.save(output / "grid_coords.npy", grid_coords)
    np.save(output / "source_node_index.npy", source_node_index)
    np.save(output / "selected_time_indices.npy", selected.astype(np.int64))
    _write_json(output / "field_names.json", FIELD_NAMES)
    _write_json(output / "static_feature_names.json", STATIC_FEATURE_NAMES)
    _write_json(output / "mask_names.json", MASK_NAMES)

    source_metadata = {}
    if metadata_path is not None and Path(metadata_path).expanduser().exists():
        source_metadata = json.loads(Path(metadata_path).expanduser().read_text(encoding="utf-8"))
    metadata = {
        **source_metadata,
        "format": "fno_grid",
        "source_format": "pinn_structured_dataset",
        "source_structured_dataset": str(structured),
        "layout": "grid_dynamic[T,C,Z,Y,X], grid_static[S,Z,Y,X]",
        "grid_resolution_zyx": list(grid_resolution),
        "interpolation": "nearest_node_from_pinn_structured_dataset",
        "selected_timesteps": int(len(selected)),
        "full_timesteps": int(len(times)),
    }
    _write_json(output / "metadata.json", metadata)
    return metadata


def _regular_grid(coords: np.ndarray, resolution: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    z_count, y_count, x_count = resolution
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    xs = np.linspace(mins[0], maxs[0], x_count, dtype=np.float32)
    ys = np.linspace(mins[1], maxs[1], y_count, dtype=np.float32)
    zs = np.linspace(mins[2], maxs[2], z_count, dtype=np.float32)
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    grid_coords = np.stack([xx, yy, zz], axis=0).astype(np.float32)
    flat = np.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], axis=1).astype(np.float32)
    return grid_coords, flat


def _nearest_indices(coords: np.ndarray, flat_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff = flat_grid[:, None, :] - coords[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    nearest_idx = np.argmin(distances, axis=1).astype(np.int64)
    nearest_dist = distances[np.arange(flat_grid.shape[0]), nearest_idx].astype(np.float32)
    return nearest_idx, nearest_dist


def _valid_grid_mask(coords: np.ndarray, nearest_dist: np.ndarray, resolution: tuple[int, int, int]) -> np.ndarray:
    span = np.maximum(coords.max(axis=0) - coords.min(axis=0), 1e-12)
    z_count, y_count, x_count = resolution
    cell = float(
        max(
            span[0] / max(x_count - 1, 1),
            span[1] / max(y_count - 1, 1),
            span[2] / max(z_count - 1, 1),
        )
    )
    return (nearest_dist <= 2.5 * cell).astype(np.float32)


def _select_time_indices(total_steps: int, max_timesteps: int | None) -> np.ndarray:
    if max_timesteps is not None and max_timesteps > 0 and total_steps > max_timesteps:
        return np.linspace(0, total_steps - 1, max_timesteps).round().astype(np.int64)
    return np.arange(total_steps, dtype=np.int64)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
