"""Dataset registry for multiple real COMSOL runs."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Optional

from .scenario import default_scenario, load_yaml, save_yaml, merged_scenario


def registry_root(registry_dir: str | Path = "datasets") -> Path:
    return Path(registry_dir)


def dataset_dir(dataset_id: str, registry_dir: str | Path = "datasets") -> Path:
    return registry_root(registry_dir) / dataset_id


def list_dataset_ids(registry_dir: str | Path = "datasets", require_processed: bool = False) -> List[str]:
    root = registry_root(registry_dir)
    if not root.exists():
        return []
    ids: List[str] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        if require_processed and not (p / "processed" / "metadata.json").exists():
            continue
        if (p / "scenario.yaml").exists() or (p / "raw").exists():
            ids.append(p.name)
    return ids


def load_scenario(dataset_id: str, registry_dir: str | Path = "datasets") -> Dict:
    path = dataset_dir(dataset_id, registry_dir) / "scenario.yaml"
    if not path.exists():
        return default_scenario(dataset_id)
    return merged_scenario(load_yaml(path), dataset_id)


def register_dataset(
    dataset_id: str,
    raw_dir: str | Path,
    mesh_file: str | Path | None = None,
    registry_dir: str | Path = "datasets",
    scenario: Optional[Dict] = None,
    *,
    copy_raw_files: bool = False,
) -> Path:
    d = dataset_dir(dataset_id, registry_dir)
    raw_target = d / "raw"
    raw_target.mkdir(parents=True, exist_ok=True)
    (d / "processed").mkdir(parents=True, exist_ok=True)
    sc = merged_scenario(scenario or default_scenario(dataset_id), dataset_id)
    sc["dataset_id"] = dataset_id
    raw_source = Path(raw_dir)
    if copy_raw_files:
        if not raw_source.exists() or not raw_source.is_dir():
            raise FileNotFoundError(f"Raw directory not found: {raw_source}")
        for item in sorted(raw_source.iterdir()):
            if not item.is_file():
                continue
            shutil.copy2(item, raw_target / item.name)
        sc.setdefault("paths", {})["raw_dir"] = "raw"
    else:
        sc.setdefault("paths", {})["raw_dir"] = str(raw_source)
    if mesh_file:
        mesh_path = Path(mesh_file)
        if copy_raw_files:
            if not mesh_path.exists() or not mesh_path.is_file():
                raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
            copied_mesh = raw_target / mesh_path.name
            if copied_mesh.resolve() != mesh_path.resolve():
                shutil.copy2(mesh_path, copied_mesh)
            sc["paths"]["mesh_file"] = f"raw/{mesh_path.name}"
            sc.setdefault("geometry", {})["mesh_file"] = mesh_path.name
        else:
            sc["paths"]["mesh_file"] = str(mesh_path)
            sc.setdefault("geometry", {})["mesh_file"] = mesh_path.name
    save_yaml(sc, d / "scenario.yaml")
    return d


def _resolve_relative(path_str: str | None, bases: list[Path]) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    if p == Path("."):
        # Empty string became '.', which caused PermissionError in parse_mphtxt.
        return None
    if p.is_absolute():
        return p
    for base in bases:
        cand = base / p
        if cand.exists():
            return cand
    return p


def resolve_raw_and_mesh(dataset_id: str, registry_dir: str | Path = "datasets") -> tuple[Path, Path | None]:
    """Resolve raw directory and optional mesh path.

    Mesh is intentionally optional.  If no .mphtxt/.mphbin file is found, the
    pipeline will build a kNN graph from CSV coordinates instead of failing.
    """
    root = registry_root(registry_dir)
    d = dataset_dir(dataset_id, registry_dir)
    sc = load_scenario(dataset_id, registry_dir)
    paths = sc.get("paths", {}) or {}
    geometry = sc.get("geometry", {}) or {}

    raw_dir = _resolve_relative(str(paths.get("raw_dir", d / "raw")), [Path.cwd(), root, d]) or (d / "raw")
    if not raw_dir.exists():
        # Prefer the canonical dataset-local raw folder over a broken YAML path.
        raw_dir = d / "raw"

    mesh_candidates_raw = [
        paths.get("mesh_file"),
        geometry.get("mesh_file"),
    ]
    mesh_file: Path | None = None
    for item in mesh_candidates_raw:
        cand = _resolve_relative(str(item) if item is not None else None, [Path.cwd(), root, d, raw_dir])
        if cand and cand.exists() and cand.is_file():
            mesh_file = cand
            break
    if mesh_file is None and raw_dir.exists():
        candidates = sorted(list(raw_dir.glob("*.mphtxt")) + list(raw_dir.glob("*.mphbin")))
        if candidates:
            mesh_file = candidates[0]
    return raw_dir, mesh_file
