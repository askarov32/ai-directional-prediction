from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TIME_SUFFIX_RE = re.compile(r"^(?P<field>.+) @ t=(?P<time>.+)$")


@dataclass(frozen=True)
class ComsolHeader:
    raw_metadata: dict[str, str]
    node_count: int
    expression_count: int
    dimension: int
    coordinate_names: tuple[str, str, str]
    field_names: tuple[str, ...]
    times: np.ndarray


@dataclass(frozen=True)
class ParsedComsolExport:
    path: Path
    header: ComsolHeader
    coordinates: np.ndarray
    values: np.ndarray

    def field(self, field_name: str) -> np.ndarray:
        try:
            field_index = self.header.field_names.index(field_name)
        except ValueError as exc:
            raise KeyError(f"Field '{field_name}' is not available in {self.path.name}") from exc
        return self.values[:, :, field_index]


def parse_comsol_csv(path: str | Path, dtype: np.dtype = np.float64) -> ParsedComsolExport:
    csv_path = Path(path).expanduser().resolve()
    rows = _read_rows(csv_path)

    metadata_rows = rows[:8]
    header_row = rows[8]
    data_rows = rows[9:]

    metadata = _parse_metadata(metadata_rows)
    coordinate_names = tuple(column.lstrip("% ").strip() for column in header_row[:3])
    field_names, times = _parse_payload_layout(header_row[3:])

    matrix = np.asarray(data_rows, dtype=dtype)
    coordinates = matrix[:, :3]
    payload = matrix[:, 3:]
    values = payload.reshape(matrix.shape[0], len(times), len(field_names))

    header = ComsolHeader(
        raw_metadata=metadata,
        node_count=int(metadata.get("Nodes", matrix.shape[0])),
        expression_count=int(metadata.get("Expressions", payload.shape[1])),
        dimension=int(metadata.get("Dimension", 0)),
        coordinate_names=coordinate_names,  # type: ignore[arg-type]
        field_names=field_names,
        times=times,
    )

    return ParsedComsolExport(
        path=csv_path,
        header=header,
        coordinates=coordinates,
        values=values,
    )


def _read_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.reader(handle))


def _parse_metadata(rows: list[list[str]]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for row in rows:
        if len(row) < 2:
            continue
        key = row[0].lstrip("% ").strip()
        metadata[key] = row[1].strip()
    return metadata


def _parse_payload_layout(columns: list[str]) -> tuple[tuple[str, ...], np.ndarray]:
    if not columns:
        raise ValueError("COMSOL export does not contain payload columns.")

    fields_by_time: dict[str, list[str]] = {}
    ordered_times: list[str] = []

    for column in columns:
        match = TIME_SUFFIX_RE.match(column.strip())
        if match is None:
            raise ValueError(f"Unsupported COMSOL payload column: {column}")
        field_name = match.group("field").strip()
        time_key = match.group("time").strip()
        if time_key not in fields_by_time:
            fields_by_time[time_key] = []
            ordered_times.append(time_key)
        fields_by_time[time_key].append(field_name)

    first_fields = fields_by_time[ordered_times[0]]
    for time_key in ordered_times[1:]:
        if fields_by_time[time_key] != first_fields:
            raise ValueError(f"Field ordering changed at time {time_key}, cannot build a stable tensor layout.")

    times = np.asarray([float(item) for item in ordered_times], dtype=np.float64)
    return tuple(first_fields), times
