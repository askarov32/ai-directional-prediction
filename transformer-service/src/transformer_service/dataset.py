from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


TIME_SUFFIX_RE = re.compile(r"^(?P<field>.+) @ t=(?P<time>.+)$")

COORD_ROUND_DECIMALS = 9

INPUT_CHANNEL_NAMES = [
    "x",
    "y",
    "z",
    "temperature_k",
    "disp_x",
    "disp_y",
    "disp_z",
    "vel_x",
    "vel_y",
    "vel_z",
    "youngs_modulus",
    "poissons_ratio",
    "density",
    "thermal_expansion",
    "thermal_conductivity",
    "heat_capacity",
]

TARGET_CHANNEL_NAMES = ["temperature_k", "disp_x", "disp_y", "disp_z"]

MATERIALS_FIELDS = ("solid.E (Pa)", "solid.nu (1)", "solid.rho (kg/m^3)", "te1.alpha_iso (1/K)")
TEMPERATURE_FIELDS = (
    "T (K)",
    "x (m)",
    "y (m)",
    "z (m)",
    "ht.k_iso (W/(m*K))",
    "ht.rho (kg/m^3)",
    "ht.Cp (J/(kg*K))",
)
DISPLACEMENT_FIELDS = (
    "u (m)",
    "v (m)",
    "w (m)",
    "ut (m/s)",
    "vt (m/s)",
    "wt (m/s)",
)


@dataclass(frozen=True)
class ParsedComsolFile:
    coords: np.ndarray
    values: np.ndarray
    field_names: tuple[str, ...]
    times: np.ndarray

    def field(self, name: str) -> np.ndarray:
        index = self.field_names.index(name)
        return self.values[:, :, index]


@dataclass(frozen=True)
class SandstoneTensors:
    state: np.ndarray
    coords: np.ndarray
    times: np.ndarray
    input_channel_names: list[str]
    target_channel_names: list[str]
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    raw_node_counts: dict[str, int]


def parse_comsol_csv(path: str | Path) -> ParsedComsolFile:
    csv_path = Path(path).expanduser().resolve()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 10:
        raise ValueError(f"COMSOL export too short: {csv_path}")

    header_row = rows[8]
    data_rows = rows[9:]

    field_names, times = _parse_payload_layout(header_row[3:])
    matrix = np.asarray(data_rows, dtype=np.float64)
    coords = matrix[:, :3]
    payload = matrix[:, 3:]
    values = payload.reshape(matrix.shape[0], len(times), len(field_names))
    return ParsedComsolFile(
        coords=coords,
        values=values,
        field_names=field_names,
        times=times,
    )


def _parse_payload_layout(columns: list[str]) -> tuple[tuple[str, ...], np.ndarray]:
    if not columns:
        raise ValueError("Empty payload columns")
    fields_by_time: dict[str, list[str]] = {}
    ordered_times: list[str] = []
    for column in columns:
        match = TIME_SUFFIX_RE.match(column.strip())
        if match is None:
            raise ValueError(f"Unrecognized column header: {column}")
        field_name = match.group("field").strip()
        time_key = match.group("time").strip()
        if time_key not in fields_by_time:
            fields_by_time[time_key] = []
            ordered_times.append(time_key)
        fields_by_time[time_key].append(field_name)

    first_fields = fields_by_time[ordered_times[0]]
    for time_key in ordered_times[1:]:
        if fields_by_time[time_key] != first_fields:
            raise ValueError(f"Field ordering changed at time {time_key}")
    times = np.asarray([float(item) for item in ordered_times], dtype=np.float64)
    return tuple(first_fields), times


def _coord_key(row: np.ndarray) -> tuple:
    return tuple(np.round(row, COORD_ROUND_DECIMALS).tolist())


def _coord_to_index(coords: np.ndarray) -> dict[tuple, int]:
    return {_coord_key(coords[i]): i for i in range(coords.shape[0])}


def build_sandstone_tensors(sandstone_dir: str | Path) -> SandstoneTensors:
    root = Path(sandstone_dir).expanduser().resolve()
    materials = parse_comsol_csv(root / "data_materials.csv")
    temperature = parse_comsol_csv(root / "data_temperature.csv")
    displacement = parse_comsol_csv(root / "data_displacement.csv")

    if not np.allclose(materials.times, temperature.times) or not np.allclose(
        materials.times, displacement.times
    ):
        raise ValueError("Time grids do not match across COMSOL exports.")
    times = materials.times.astype(np.float64)

    disp_lookup = _coord_to_index(displacement.coords)
    indices_full: list[int] = []
    indices_disp: list[int] = []
    for i in range(materials.coords.shape[0]):
        key = _coord_key(materials.coords[i])
        match = disp_lookup.get(key)
        if match is not None:
            indices_full.append(i)
            indices_disp.append(match)
    if not indices_full:
        raise ValueError("Intersection of node coordinates is empty.")
    indices_full_arr = np.asarray(indices_full, dtype=np.int64)
    indices_disp_arr = np.asarray(indices_disp, dtype=np.int64)

    if not np.allclose(
        materials.coords[indices_full_arr], displacement.coords[indices_disp_arr]
    ):
        raise ValueError(
            "Intersection mapping produced misaligned coordinates — check rounding tolerance."
        )

    coords_common = materials.coords[indices_full_arr].astype(np.float32)
    n_common = coords_common.shape[0]
    n_times = times.shape[0]

    temperature_field = temperature.field(TEMPERATURE_FIELDS[0])[indices_full_arr]
    thermal_k = temperature.field(TEMPERATURE_FIELDS[4])[indices_full_arr]
    thermal_cp = temperature.field(TEMPERATURE_FIELDS[6])[indices_full_arr]

    materials_static = np.stack(
        [materials.field(name)[indices_full_arr, 0] for name in MATERIALS_FIELDS],
        axis=-1,
    )

    disp_dynamic = np.stack(
        [displacement.field(name)[indices_disp_arr] for name in DISPLACEMENT_FIELDS],
        axis=-1,
    )

    coords_dynamic = np.broadcast_to(
        coords_common[:, None, :], (n_common, n_times, 3)
    )
    materials_full = np.broadcast_to(
        materials_static[:, None, :], (n_common, n_times, materials_static.shape[1])
    )
    thermal_static_full = np.broadcast_to(
        np.stack([thermal_k[:, 0], thermal_cp[:, 0]], axis=-1)[:, None, :],
        (n_common, n_times, 2),
    )

    state = np.concatenate(
        [
            coords_dynamic,
            temperature_field[..., None],
            disp_dynamic,
            materials_full,
            thermal_static_full,
        ],
        axis=-1,
    ).astype(np.float32)

    if state.shape[-1] != len(INPUT_CHANNEL_NAMES):
        raise ValueError(
            f"State channel count mismatch: got {state.shape[-1]}, expected {len(INPUT_CHANNEL_NAMES)}"
        )

    target_indices = [INPUT_CHANNEL_NAMES.index(name) for name in TARGET_CHANNEL_NAMES]

    flat_in = state.reshape(-1, state.shape[-1])
    input_mean = flat_in.mean(axis=0)
    input_std = flat_in.std(axis=0)
    input_std = np.where(input_std < 1e-8, 1.0, input_std).astype(np.float32)
    target_mean = input_mean[target_indices].astype(np.float32)
    target_std = input_std[target_indices].astype(np.float32)

    return SandstoneTensors(
        state=state,
        coords=coords_common,
        times=times.astype(np.float32),
        input_channel_names=list(INPUT_CHANNEL_NAMES),
        target_channel_names=list(TARGET_CHANNEL_NAMES),
        input_mean=input_mean.astype(np.float32),
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
        raw_node_counts={
            "materials": int(materials.coords.shape[0]),
            "temperature": int(temperature.coords.shape[0]),
            "displacement": int(displacement.coords.shape[0]),
            "intersection": int(n_common),
        },
    )


def save_sandstone_artifacts(tensors: SandstoneTensors, output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    pairs_path = out / "pairs.npz"
    np.savez_compressed(
        pairs_path,
        state=tensors.state,
        coords=tensors.coords,
        times=tensors.times,
        input_channel_names=np.asarray(tensors.input_channel_names, dtype="<U32"),
        target_channel_names=np.asarray(tensors.target_channel_names, dtype="<U32"),
        input_mean=tensors.input_mean,
        input_std=tensors.input_std,
        target_mean=tensors.target_mean,
        target_std=tensors.target_std,
    )

    scalers_path = out / "scalers.json"
    scalers_path.write_text(
        json.dumps(
            {
                "input_channel_names": tensors.input_channel_names,
                "input_scaler": {
                    "mean": tensors.input_mean.tolist(),
                    "std": tensors.input_std.tolist(),
                },
                "target_channel_names": tensors.target_channel_names,
                "target_scaler": {
                    "mean": tensors.target_mean.tolist(),
                    "std": tensors.target_std.tolist(),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    metadata_path = out / "dataset_metadata.json"
    n_pairs = int(tensors.state.shape[1]) - 1
    train_size = int(n_pairs * 0.8)
    metadata_path.write_text(
        json.dumps(
            {
                "node_count": int(tensors.state.shape[0]),
                "time_steps": int(tensors.state.shape[1]),
                "input_channels": int(tensors.state.shape[2]),
                "target_channels": len(tensors.target_channel_names),
                "raw_node_counts": tensors.raw_node_counts,
                "n_pairs": n_pairs,
                "train_size": train_size,
                "val_size": n_pairs - train_size,
                "time_start": float(tensors.times[0]),
                "time_end": float(tensors.times[-1]),
                "time_step": float(tensors.times[1] - tensors.times[0])
                if len(tensors.times) > 1
                else 0.0,
                "input_channel_names": tensors.input_channel_names,
                "target_channel_names": tensors.target_channel_names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"pairs": pairs_path, "scalers": scalers_path, "metadata": metadata_path}


@dataclass(frozen=True)
class LoadedPairsBundle:
    state: np.ndarray
    coords: np.ndarray
    times: np.ndarray
    input_channel_names: list[str]
    target_channel_names: list[str]
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray


def load_pairs_bundle(dataset_path: str | Path) -> LoadedPairsBundle:
    payload = np.load(Path(dataset_path).expanduser().resolve())
    return LoadedPairsBundle(
        state=payload["state"].astype(np.float32),
        coords=payload["coords"].astype(np.float32),
        times=payload["times"].astype(np.float32),
        input_channel_names=[str(s) for s in payload["input_channel_names"].tolist()],
        target_channel_names=[str(s) for s in payload["target_channel_names"].tolist()],
        input_mean=payload["input_mean"].astype(np.float32),
        input_std=payload["input_std"].astype(np.float32),
        target_mean=payload["target_mean"].astype(np.float32),
        target_std=payload["target_std"].astype(np.float32),
    )


class AutoregressivePairsDataset(Dataset):
    def __init__(
        self,
        bundle: LoadedPairsBundle,
        pair_indices: list[int],
        n_tokens: int | None = None,
        seed: int = 42,
    ) -> None:
        if not pair_indices:
            raise ValueError("pair_indices must not be empty")
        self._bundle = bundle
        self._pair_indices = list(pair_indices)
        self._target_channel_indices = [
            bundle.input_channel_names.index(name) for name in bundle.target_channel_names
        ]
        self._input_mean = bundle.input_mean
        self._input_std = bundle.input_std
        self._target_mean = bundle.target_mean
        self._target_std = bundle.target_std
        self._coords_norm = (bundle.coords - bundle.input_mean[:3]) / bundle.input_std[:3]
        node_count = bundle.state.shape[0]
        if n_tokens is not None and n_tokens > 0 and n_tokens < node_count:
            self._n_tokens: int | None = int(n_tokens)
        else:
            self._n_tokens = None
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._pair_indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        k = self._pair_indices[index]
        state_k = self._bundle.state[:, k, :]
        state_next = self._bundle.state[:, k + 1, :]
        node_count = state_k.shape[0]
        if self._n_tokens is not None:
            token_indices = self._rng.choice(node_count, size=self._n_tokens, replace=False)
        else:
            token_indices = np.arange(node_count)
        input_tokens = (state_k[token_indices] - self._input_mean) / self._input_std
        target_raw = state_next[token_indices][:, self._target_channel_indices]
        target_norm = (target_raw - self._target_mean) / self._target_std
        query_coords = self._coords_norm[token_indices]
        return {
            "input_tokens": torch.from_numpy(input_tokens.astype(np.float32)),
            "query_coords": torch.from_numpy(query_coords.astype(np.float32)),
            "target": torch.from_numpy(target_norm.astype(np.float32)),
            "pair_index": torch.tensor(int(k), dtype=torch.long),
        }


def build_train_val_split(bundle: LoadedPairsBundle, train_fraction: float = 0.8) -> tuple[list[int], list[int]]:
    n_pairs = int(bundle.state.shape[1]) - 1
    if n_pairs <= 0:
        raise ValueError("Bundle has insufficient time steps for autoregressive pairs.")
    train_size = max(1, int(n_pairs * train_fraction))
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, n_pairs))
    if not val_indices:
        val_indices = train_indices[-1:]
    return train_indices, val_indices
