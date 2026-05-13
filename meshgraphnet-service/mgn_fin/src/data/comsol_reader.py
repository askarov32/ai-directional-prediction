"""COMSOL CSV reader for real exported datasets.

This module is intentionally strict about real COMSOL data, but tolerant to the
formats COMSOL often writes:
- metadata/comment lines start with "%";
- the actual header may also start with "% X,Y,Z,...";
- different exports may contain different node counts;
- material/helper fields may be exported together with physical state fields.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


TIME_COL_RE = re.compile(r"^(.+?)\s*\((.*?)\)\s*@\s*t\s*=\s*([\dEe+\-.]+)\s*$", re.IGNORECASE)

# These fields are not part of the dynamic state that the network should learn to
# predict. They are exported by COMSOL as helper/material fields and are moved to
# scenario/material metadata instead.
DEFAULT_EXCLUDED_DYNAMIC_FIELDS = {
    "x", "y", "z",
    "ht.k_iso", "ht.rho", "ht.cp",
    "solid.e", "solid.nu", "solid.rho", "solid.alpha_iso", "solid.alpha_iso", "te1.alpha_iso",
}
DEFAULT_EXCLUDED_FILE_HINTS = {"material", "materials"}
ALLOWED_RAW_CSV_PREFIXES = (
    "data_displacement",
    "data_temperature",
    "data_strain",
    "data_stress",
    "data_materials",
)


def _clean_header_candidate(line: str) -> str:
    line = line.strip()
    if line.startswith("%"):
        line = line[1:].strip()
    return line


def _find_header(lines: List[str]) -> int:
    """Find a COMSOL header row.

    COMSOL commonly writes the header as a comment line: "% X,Y,Z,...".
    Older parsers often skipped it, which broke real exports. Here we explicitly
    allow a leading "%" as long as the cleaned line contains X and Y columns.
    """
    for i, line in enumerate(lines):
        cleaned = _clean_header_candidate(line)
        if not cleaned:
            continue
        # Real coordinate header should begin with X,Y or X  Y, not merely
        # contain words such as "X component" in a Description metadata line.
        if re.match(r"^X\s*(?:,|\s+)\s*Y\b", cleaned, re.IGNORECASE):
            return i
    raise ValueError("Cannot find COMSOL header row with X and Y columns.")


def _split_header(header: str) -> List[str]:
    header = _clean_header_candidate(header)
    if "," in header:
        return [c.strip() for c in header.split(",")]
    pattern = re.compile(
        r"(?:X|Y|Z)"
        r"|[^\s].*?\([^)]*\)\s*@\s*t\s*=\s*[\dEe+\-.]+(?=\s{2,}|$)"
        r"|[^\s].*?\([^)]*\)(?=\s{2,}|$)",
        re.IGNORECASE,
    )
    cols = [c.strip() for c in pattern.findall(header)]
    if len(cols) < 2:
        cols = [c.strip() for c in re.split(r"\s{2,}", header.strip()) if c.strip()]
    return cols


def parse_comsol_csv(filepath: str | Path) -> pd.DataFrame:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    # Stream only the metadata/header part. Full read_text() is too slow and
    # memory-heavy for real COMSOL exports with hundreds of columns.
    header_idx = None
    header_line = None
    with filepath.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            cleaned = _clean_header_candidate(line)
            if cleaned and re.match(r"^X\s*(?:,|\s+)\s*Y\b", cleaned, re.IGNORECASE):
                header_idx = i
                header_line = cleaned
                break
    if header_idx is None or header_line is None:
        raise ValueError(f"Cannot find COMSOL header row with X and Y columns in {filepath.name}")

    col_names = _split_header(header_line)

    if "," in header_line:
        # Real COMSOL CSV: metadata lines start with %, header is % X,Y,Z,...
        # We skip the header and assign cleaned names manually.
        df = pd.read_csv(
            filepath,
            header=None,
            skiprows=header_idx + 1,
            comment="%",
            low_memory=False,
        )
        # Assign names after reading to avoid pandas duplicate-name validation bugs
        # on very wide COMSOL headers.
        if df.shape[1] > len(col_names):
            df = df.iloc[:, :len(col_names)]
        elif df.shape[1] < len(col_names):
            for j in range(df.shape[1], len(col_names)):
                df[j] = np.nan
        df.columns = col_names
        df = df.apply(pd.to_numeric, errors="coerce")
    else:
        # Fallback for whitespace-separated exports. These are usually smaller;
        # parse manually with numeric regex.
        data_lines = []
        with filepath.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i <= header_idx:
                    continue
                if line.strip() and not line.lstrip().startswith("%"):
                    data_lines.append(line)
        if not data_lines:
            raise ValueError(f"No data rows found in {filepath.name}")
        num_re = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
        rows = [num_re.findall(line) for line in data_lines]
        n_cols = len(col_names)
        fixed = []
        for r in rows:
            if len(r) >= n_cols:
                fixed.append(r[:n_cols])
            elif r:
                fixed.append(r + ["nan"] * (n_cols - len(r)))
        if not fixed:
            raise ValueError(f"Cannot parse numeric rows in {filepath.name}")
        df = pd.DataFrame(fixed, columns=col_names).apply(pd.to_numeric, errors="coerce")

    coord_cols = coordinate_columns(df)
    if len(coord_cols) < 2:
        raise ValueError(f"No X/Y coordinate columns in {filepath.name}. Columns: {list(df.columns)[:10]}")
    return df


def coordinate_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if str(c).strip().upper() in {"X", "Y", "Z"}]


def coords_from_df(df: pd.DataFrame) -> np.ndarray:
    cols = coordinate_columns(df)
    coords = df[cols].to_numpy(dtype=np.float32)
    if coords.shape[1] == 2:
        coords = np.column_stack([coords, np.zeros(len(coords), dtype=np.float32)])
    return coords[:, :3].astype(np.float32)


def parse_time_columns(df: pd.DataFrame) -> Tuple[List[Dict], List[float]]:
    info = []
    coord_cols = set(coordinate_columns(df))
    for col in df.columns:
        if col in coord_cols:
            continue
        m = TIME_COL_RE.match(str(col).strip())
        if not m:
            continue
        field = m.group(1).strip().lower().replace(" ", "_")
        unit = m.group(2).strip()
        try:
            time = float(m.group(3))
        except ValueError:
            continue
        info.append({"field": field, "unit": unit, "time": time, "col": col})
    return info, sorted({x["time"] for x in info})


def load_raw_csvs(raw_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """Load only real COMSOL physics CSV files.

    Mesh/helper CSV files such as ``basalt_mesh.csv`` must not be parsed as
    COMSOL field exports.  The accepted files are the project contract files:
    data_displacement*.csv, data_temperature*.csv, data_strain*.csv,
    data_stress*.csv and data_materials*.csv.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
    all_csvs = sorted(raw_dir.glob("*.csv"))
    csvs = [f for f in all_csvs if f.stem.lower().startswith(ALLOWED_RAW_CSV_PREFIXES)]
    ignored = [f.name for f in all_csvs if f not in csvs]
    if ignored:
        warnings.warn(
            "Ignoring non-COMSOL-field CSV files in raw/: " + ", ".join(ignored[:20])
            + (" ..." if len(ignored) > 20 else "")
        )
    if not csvs:
        raise FileNotFoundError(
            f"No supported COMSOL CSV files found in {raw_dir}. Expected prefixes: {ALLOWED_RAW_CSV_PREFIXES}"
        )
    out = {}
    for f in csvs:
        out[f.stem] = parse_comsol_csv(f)
    return out


def _is_excluded_file(file_key: str, excluded_file_hints: Iterable[str]) -> bool:
    k = file_key.lower()
    return any(h.lower() in k for h in excluded_file_hints)


def _is_excluded_field(field: str, excluded_fields: Iterable[str]) -> bool:
    f = field.lower().strip()
    return f in {x.lower().strip() for x in excluded_fields}


def dynamic_csv_dict(csv_dict: Dict[str, pd.DataFrame], excluded_file_hints: Iterable[str] = DEFAULT_EXCLUDED_FILE_HINTS) -> Dict[str, pd.DataFrame]:
    return {k: v for k, v in csv_dict.items() if not _is_excluded_file(k, excluded_file_hints)}


def choose_reference_coordinates(csv_dict: Dict[str, pd.DataFrame]) -> Tuple[str, np.ndarray]:
    """Choose reference nodes for a consistent graph.

    If COMSOL exports different node counts per physics, the safest common base
    is usually the smallest dynamic field export. In the uploaded dataset this is
    displacement: 4448 nodes, while other fields have 4743 nodes. Larger exports
    are then reordered/mapped to those 4448 nodes by exact/nearest coordinates.
    """
    if not csv_dict:
        raise ValueError("No CSV files available for coordinate reference.")
    best_key = min(csv_dict.keys(), key=lambda k: len(csv_dict[k]))
    return best_key, coords_from_df(csv_dict[best_key])


def align_coordinates(csv_dict: Dict[str, pd.DataFrame], tol: float = 1e-7) -> np.ndarray:
    dyn = dynamic_csv_dict(csv_dict)
    ref_key, coords = choose_reference_coordinates(dyn or csv_dict)
    for key, df in (dyn or csv_dict).items():
        other = coords_from_df(df)
        if len(other) == len(coords):
            diff = float(np.nanmax(np.abs(coords - other)))
            if diff <= tol:
                continue
        nbrs = NearestNeighbors(n_neighbors=1).fit(other)
        dist, _ = nbrs.kneighbors(coords)
        max_dist = float(dist.max()) if len(dist) else 0.0
        if max_dist > tol:
            warnings.warn(
                f"Coordinates in {key} are not an exact match to reference {ref_key}; max nearest distance={max_dist:.6g}. "
                "Fields will be mapped by nearest coordinates."
            )
    return coords


def _index_to_reference(source_coords: np.ndarray, ref_coords: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    if len(source_coords) == len(ref_coords) and np.nanmax(np.abs(source_coords - ref_coords)) <= tol:
        return np.arange(len(ref_coords), dtype=np.int64)
    nbrs = NearestNeighbors(n_neighbors=1).fit(source_coords)
    dist, idx = nbrs.kneighbors(ref_coords)
    max_dist = float(dist.max()) if len(dist) else 0.0
    if max_dist > tol:
        warnings.warn(f"Nearest-node mapping max distance is {max_dist:.6g}; check that CSV files use the same geometry.")
    return idx[:, 0].astype(np.int64)


def build_state_tensor(
    csv_dict: Dict[str, pd.DataFrame],
    reference_coords: np.ndarray | None = None,
    excluded_fields: Iterable[str] = DEFAULT_EXCLUDED_DYNAMIC_FIELDS,
    excluded_file_hints: Iterable[str] = DEFAULT_EXCLUDED_FILE_HINTS,
) -> Tuple[np.ndarray, List[str], List[float], Dict[str, str]]:
    """Build [T, N, F] state tensor from all COMSOL wide CSV files.

    Material/helper files are excluded from the dynamic state. Larger CSV exports
    are mapped to the reference node set by coordinates, instead of being sliced
    by row order.
    """
    dyn_csvs = dynamic_csv_dict(csv_dict, excluded_file_hints)
    if not dyn_csvs:
        raise ValueError("No dynamic CSV files found after excluding material files.")

    ref_key, ref_coords = choose_reference_coordinates(dyn_csvs)
    if reference_coords is not None:
        ref_coords = np.asarray(reference_coords, dtype=np.float32)

    all_infos = []
    units: Dict[str, str] = {}
    for key, df in dyn_csvs.items():
        info, _ = parse_time_columns(df)
        for item in info:
            field = item["field"]
            if _is_excluded_field(field, excluded_fields):
                continue
            item = dict(item)
            item["file_key"] = key
            all_infos.append(item)
            units.setdefault(field, item["unit"])
    if not all_infos:
        raise ValueError("No time-dependent COMSOL state columns found after filtering helper/material fields.")

    times = sorted({x["time"] for x in all_infos})
    fields = sorted({x["field"] for x in all_infos})
    N = len(ref_coords)
    arr = np.full((len(times), N, len(fields)), np.nan, dtype=np.float32)
    t_to_i = {t: i for i, t in enumerate(times)}
    f_to_i = {f: i for i, f in enumerate(fields)}

    index_cache: Dict[str, np.ndarray] = {}
    for key, df in dyn_csvs.items():
        index_cache[key] = _index_to_reference(coords_from_df(df), ref_coords)

    for item in all_infos:
        df = dyn_csvs[item["file_key"]]
        idx = index_cache[item["file_key"]]
        values = df[item["col"]].to_numpy(dtype=np.float32)
        arr[t_to_i[item["time"]], :, f_to_i[item["field"]]] = values[idx]

    # Fill missing values per field with last valid / mean / zero.
    for fi, field in enumerate(fields):
        v = arr[:, :, fi]
        if np.isnan(v).all():
            arr[:, :, fi] = 0.0
            continue
        mean = float(np.nanmean(v))
        arr[:, :, fi] = np.nan_to_num(v, nan=mean, posinf=mean, neginf=mean)
    return arr, fields, times, units


def extract_material_metadata(csv_dict: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """Extract constant material/scenario parameters from COMSOL exports."""
    candidates = {
        "young_modulus": ["solid.e"],
        "poisson_ratio": ["solid.nu"],
        "density": ["solid.rho", "ht.rho"],
        "thermal_expansion": ["solid.alpha_iso", "te1.alpha_iso"],
        "thermal_conductivity": ["ht.k_iso"],
        "heat_capacity": ["ht.cp"],
    }
    found: Dict[str, float] = {}
    for key, df in csv_dict.items():
        infos, _ = parse_time_columns(df)
        by_field = {}
        for info in infos:
            by_field.setdefault(info["field"].lower(), []).append(info["col"])
        for out_key, names in candidates.items():
            if out_key in found and found[out_key] not in {0.0, None}:
                continue
            for name in names:
                cols = by_field.get(name.lower())
                if not cols:
                    continue
                values = []
                # sampling the first few time columns is enough for constants
                for col in cols[: min(3, len(cols))]:
                    v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
                    if np.isfinite(v).any():
                        values.append(float(np.nanmean(v)))
                if values:
                    found[out_key] = float(np.nanmean(values))
                    break
    return found


def infer_temperature_source(csv_dict: Dict[str, pd.DataFrame], threshold_fraction: float = 0.2) -> Dict[str, object]:
    """Infer source/background temperature and approximate hot region from T at t=0."""
    for key, df in csv_dict.items():
        infos, _ = parse_time_columns(df)
        t_cols = [x for x in infos if x["field"].lower() in {"t", "temperature"}]
        if not t_cols:
            continue
        # Prefer t=0 if present.
        item = sorted(t_cols, key=lambda x: abs(float(x["time"])))[0]
        temp = pd.to_numeric(df[item["col"]], errors="coerce").to_numpy(dtype=np.float64)
        coords = coords_from_df(df).astype(np.float64)
        bg = float(np.nanmedian(temp))
        tmax = float(np.nanmax(temp))
        out: Dict[str, object] = {"background_temperature": bg, "initial_temperature": tmax}
        if np.isfinite(tmax) and np.isfinite(bg) and tmax > bg:
            threshold = bg + threshold_fraction * (tmax - bg)
            mask = temp >= threshold
            if np.any(mask):
                hot_coords = coords[mask]
                center = hot_coords.mean(axis=0)
                radius = float(np.percentile(np.linalg.norm(hot_coords - center.reshape(1, 3), axis=1), 95))
                out["center"] = [float(x) for x in center[:3]]
                out["radius"] = radius
        return out
    return {}


def extract_node_static_material_fields(
    csv_dict: Dict[str, pd.DataFrame],
    reference_coords: np.ndarray,
    fields_to_extract: Iterable[str] = ("solid.e", "solid.nu", "solid.rho", "solid.alpha_iso", "te1.alpha_iso", "ht.k_iso", "ht.rho", "ht.cp"),
) -> Tuple[np.ndarray, List[str], Dict[str, str]]:
    """Extract spatially varying material/helper fields as node-static features.

    COMSOL exports material properties per mesh point. For heterogeneous systems
    such as heated steel rod + geological medium, using only a global average is
    physically weak. These fields are therefore added to node_static so the model
    knows which nodes belong to which material/thermal domain.
    """
    ref = np.asarray(reference_coords, dtype=np.float32)
    wanted = {f.lower() for f in fields_to_extract}
    arrays: List[np.ndarray] = []
    names: List[str] = []
    units: Dict[str, str] = {}
    seen = set()
    for key, df in csv_dict.items():
        infos, _ = parse_time_columns(df)
        # group by field and choose the first available/earliest time column
        by_field: Dict[str, List[Dict]] = {}
        for info in infos:
            f = info["field"].lower()
            if f in wanted:
                by_field.setdefault(f, []).append(info)
        if not by_field:
            continue
        idx = _index_to_reference(coords_from_df(df), ref)
        for f, items in sorted(by_field.items()):
            if f in seen:
                continue
            item = sorted(items, key=lambda x: abs(float(x["time"])))[0]
            values = pd.to_numeric(df[item["col"]], errors="coerce").to_numpy(dtype=np.float32)[idx]
            if not np.isfinite(values).any():
                continue
            mean = float(np.nanmean(values))
            values = np.nan_to_num(values, nan=mean, posinf=mean, neginf=mean).astype(np.float32)
            name = "node_" + f.replace(".", "_")
            arrays.append(values.reshape(-1, 1))
            names.append(name)
            units[name] = item.get("unit", "")
            seen.add(f)
    if not arrays:
        return np.zeros((len(ref), 0), dtype=np.float32), [], {}
    return np.concatenate(arrays, axis=1).astype(np.float32), names, units
