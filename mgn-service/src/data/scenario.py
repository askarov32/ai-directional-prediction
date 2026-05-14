"""Scenario metadata utilities for Conditional MeshGraphNet.

The project historically used a ``source`` block for the heated-rod case.  The
commercial schema uses a richer ``scenario`` block with multiple scenario types.
This module keeps both formats compatible and exposes a fixed numeric feature
schema so processed datasets from different rocks/scenarios can be compared and
trained together when their dynamic fields match.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

SUPPORTED_SCENARIOS = ("heated_rod", "impact", "side_pressure", "building_load")
SIDE_TO_ID = {"x_min": 0.0, "x_max": 1.0, "y_min": 2.0, "y_max": 3.0, "z_min": 4.0, "z_max": 5.0}
LOAD_TYPE_TO_ID = {"static": 0.0, "ramp": 1.0, "pulse": 2.0}


def stable_hash01(value: Any) -> float:
    """Stable category encoding in [0, 1]. Keeps schemas numeric without a registry DB."""
    s = str(value).strip().lower()
    if not s:
        return 0.0
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _as_vec3(value: Any, default: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> List[float]:
    if value is None:
        return list(default)
    if not isinstance(value, (list, tuple, np.ndarray)):
        return list(default)
    vals = [_as_float(v, 0.0) for v in list(value)]
    return (vals + list(default))[:3]


def _as_vec2(value: Any, default: Tuple[float, float] = (0.0, 0.0)) -> List[float]:
    if value is None:
        return list(default)
    if not isinstance(value, (list, tuple, np.ndarray)):
        return list(default)
    vals = [_as_float(v, 0.0) for v in list(value)]
    return (vals + list(default))[:2]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def default_scenario(dataset_id: str = "dataset") -> Dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "rock_type": "unknown",
        "physics": {
            "type": "thermoelastic_wave",
            "heat_transfer": True,
            "solid_mechanics": True,
            "thermal_expansion": True,
        },
        "geometry": {"mesh_file": "", "dimension": 3, "graph_source": "mesh"},
        "scenario": {
            "type": "heated_rod",
            "initial_temperature": 0.0,
            "background_temperature": 0.0,
            "source_center": [0.0, 0.0, 0.0],
            "source_radius": 0.0,
        },
        "material": {
            "source": "data_materials.csv",
            "has_nodewise_materials": True,
            "young_modulus": 0.0,
            "poisson_ratio": 0.0,
            "density": 0.0,
            "thermal_expansion": 0.0,
            "thermal_conductivity": 0.0,
            "heat_capacity": 0.0,
        },
        "time": {"start": 0.0, "end": 0.0, "step": 0.0},
        "training": {"target_mode": "delta", "train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
        "boundary_conditions": {"mechanical": "unknown", "thermal": "unknown"},
    }


def deep_update(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a or {})
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def normalize_scenario_schema(scenario: Dict[str, Any], dataset_id: str | None = None) -> Dict[str, Any]:
    """Normalize legacy and commercial scenario.yaml schemas.

    Supported input variants:
    - legacy: ``source: {type: heated_rod, center, radius, ...}``
    - commercial: ``scenario: {type: heated_rod|impact|side_pressure|building_load, ...}``
    The returned dict always contains both ``scenario`` and legacy-compatible
    ``source`` for older inference code paths.
    """
    sc = deep_update(default_scenario(dataset_id or scenario.get("dataset_id", "dataset")), scenario or {})
    if dataset_id:
        sc["dataset_id"] = dataset_id

    legacy_source = dict(sc.get("source", {}) or {})
    s = dict(sc.get("scenario", {}) or {})
    if legacy_source:
        # Legacy source block wins where commercial values are absent or still
        # default zero placeholders. This keeps old scenario.yaml files useful.
        if not s.get("type") or s.get("type") == "heated_rod":
            s["type"] = legacy_source.get("type", s.get("type", "heated_rod"))
        if "center" in legacy_source and ("source_center" not in s or s.get("source_center") in ([0, 0, 0], [0.0, 0.0, 0.0], None)):
            s["source_center"] = legacy_source.get("center")
        if "radius" in legacy_source and _as_float(s.get("source_radius", 0.0)) == 0.0:
            s["source_radius"] = legacy_source.get("radius")
        for key in ["initial_temperature", "background_temperature"]:
            if key in legacy_source and _as_float(s.get(key, 0.0)) == 0.0:
                s[key] = legacy_source.get(key)

    stype = str(s.get("type", "heated_rod") or "heated_rod").strip().lower()
    if stype not in SUPPORTED_SCENARIOS:
        # Unknown custom scenarios are allowed, but encoded via hashes.  They use
        # generic zero-valued known-parameter slots unless the user extends code.
        stype = str(s.get("type", "custom")).strip().lower() or "custom"
    s["type"] = stype

    # Canonical defaults per scenario type.
    if stype == "heated_rod":
        s["source_center"] = _as_vec3(s.get("source_center", s.get("center", [0.0, 0.0, 0.0])))
        s["source_radius"] = _as_float(s.get("source_radius", s.get("radius", 0.0)))
        s["initial_temperature"] = _as_float(s.get("initial_temperature", 0.0))
        s["background_temperature"] = _as_float(s.get("background_temperature", 0.0))
    elif stype == "impact":
        s["impact_location"] = _as_vec3(s.get("impact_location", [0.0, 0.0, 0.0]))
        s["impact_direction"] = _as_vec3(s.get("impact_direction", [0.0, 0.0, -1.0]))
        s["impact_radius"] = _as_float(s.get("impact_radius", 0.0))
        s["impact_force"] = _as_float(s.get("impact_force", 0.0))
        s["impact_duration"] = _as_float(s.get("impact_duration", 0.0))
        s["initial_velocity"] = _as_float(s.get("initial_velocity", 0.0))
    elif stype == "side_pressure":
        s["pressure_side"] = str(s.get("pressure_side", "x_min"))
        s["pressure_value"] = _as_float(s.get("pressure_value", 0.0))
        s["pressure_duration"] = _as_float(s.get("pressure_duration", 0.0))
        s["loading_profile"] = str(s.get("loading_profile", "static"))
    elif stype == "building_load":
        s["load_area_center"] = _as_vec3(s.get("load_area_center", [0.0, 0.0, 0.0]))
        s["load_area_size"] = _as_vec2(s.get("load_area_size", [0.0, 0.0]))
        s["load_value"] = _as_float(s.get("load_value", 0.0))
        s["load_type"] = str(s.get("load_type", "static"))
        s["duration"] = _as_float(s.get("duration", 0.0))
        s["foundation_geometry"] = str(s.get("foundation_geometry", "rectangular"))

    sc["scenario"] = s
    # Legacy-compatible source block used by old inference/plot code.
    sc["source"] = {
        "type": s.get("type", "heated_rod"),
        "initial_temperature": s.get("initial_temperature", legacy_source.get("initial_temperature", 0.0)),
        "background_temperature": s.get("background_temperature", legacy_source.get("background_temperature", 0.0)),
        "center": s.get("source_center", s.get("impact_location", s.get("load_area_center", legacy_source.get("center", [0, 0, 0])))),
        "radius": s.get("source_radius", s.get("impact_radius", legacy_source.get("radius", 0.0))),
    }
    return sc


def merged_scenario(user_scenario: Dict[str, Any], dataset_id: str | None = None) -> Dict[str, Any]:
    return normalize_scenario_schema(user_scenario or {}, dataset_id=dataset_id)


def scenario_feature_vector(scenario: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    """Convert scenario.yaml into a fixed numeric vector."""
    sc = normalize_scenario_schema(scenario)
    material = sc.get("material", {}) or {}
    physics = sc.get("physics", {}) or {}
    time = sc.get("time", {}) or {}
    bc = sc.get("boundary_conditions", {}) or {}
    s = sc.get("scenario", {}) or {}
    stype = str(s.get("type", "custom")).lower()

    source_center = _as_vec3(s.get("source_center", sc.get("source", {}).get("center", [0.0, 0.0, 0.0])))
    impact_location = _as_vec3(s.get("impact_location", source_center))
    impact_direction = _as_vec3(s.get("impact_direction", [0.0, 0.0, -1.0]))
    load_center = _as_vec3(s.get("load_area_center", source_center))
    load_size = _as_vec2(s.get("load_area_size", [0.0, 0.0]))

    names = [
        "mat_young_modulus", "mat_poisson_ratio", "mat_density", "mat_thermal_expansion",
        "mat_thermal_conductivity", "mat_heat_capacity", "time_dt", "rock_type_hash",
        "physics_type_hash", "scenario_type_hash", "bc_mechanical_hash", "bc_thermal_hash",
        "flag_heat_transfer", "flag_solid_mechanics", "flag_thermal_expansion",
        "scenario_is_heated_rod", "scenario_is_impact", "scenario_is_side_pressure", "scenario_is_building_load",
        "heated_initial_temperature", "heated_background_temperature", "heated_source_radius",
        "heated_source_center_x", "heated_source_center_y", "heated_source_center_z",
        "impact_location_x", "impact_location_y", "impact_location_z", "impact_radius",
        "impact_force", "impact_duration", "impact_direction_x", "impact_direction_y", "impact_direction_z",
        "impact_initial_velocity", "side_pressure_value", "side_pressure_duration", "side_pressure_side_id",
        "side_pressure_loading_profile_hash", "building_load_center_x", "building_load_center_y", "building_load_center_z",
        "building_load_size_x", "building_load_size_y", "building_load_value", "building_load_type_id",
        "building_load_duration", "foundation_geometry_hash",
    ]
    values = [
        _as_float(material.get("young_modulus", 0.0)),
        _as_float(material.get("poisson_ratio", 0.0)),
        _as_float(material.get("density", 0.0)),
        _as_float(material.get("thermal_expansion", 0.0)),
        _as_float(material.get("thermal_conductivity", 0.0)),
        _as_float(material.get("heat_capacity", 0.0)),
        _as_float(time.get("step", 0.0)),
        stable_hash01(sc.get("rock_type", "unknown")),
        stable_hash01(physics.get("type", "unknown")),
        stable_hash01(stype),
        stable_hash01(bc.get("mechanical", "unknown")),
        stable_hash01(bc.get("thermal", "unknown")),
        1.0 if physics.get("heat_transfer", False) else 0.0,
        1.0 if physics.get("solid_mechanics", False) else 0.0,
        1.0 if physics.get("thermal_expansion", False) else 0.0,
        1.0 if stype == "heated_rod" else 0.0,
        1.0 if stype == "impact" else 0.0,
        1.0 if stype == "side_pressure" else 0.0,
        1.0 if stype == "building_load" else 0.0,
        _as_float(s.get("initial_temperature", 0.0)),
        _as_float(s.get("background_temperature", 0.0)),
        _as_float(s.get("source_radius", 0.0)),
        source_center[0], source_center[1], source_center[2],
        impact_location[0], impact_location[1], impact_location[2],
        _as_float(s.get("impact_radius", 0.0)),
        _as_float(s.get("impact_force", 0.0)),
        _as_float(s.get("impact_duration", 0.0)),
        impact_direction[0], impact_direction[1], impact_direction[2],
        _as_float(s.get("initial_velocity", 0.0)),
        _as_float(s.get("pressure_value", 0.0)),
        _as_float(s.get("pressure_duration", 0.0)),
        SIDE_TO_ID.get(str(s.get("pressure_side", "")).lower(), -1.0),
        stable_hash01(s.get("loading_profile", "unknown")),
        load_center[0], load_center[1], load_center[2],
        load_size[0], load_size[1],
        _as_float(s.get("load_value", 0.0)),
        LOAD_TYPE_TO_ID.get(str(s.get("load_type", "")).lower(), -1.0),
        _as_float(s.get("duration", 0.0)),
        stable_hash01(s.get("foundation_geometry", "unknown")),
    ]
    return np.asarray(values, dtype=np.float32), names


def source_center_radius_temperature(scenario: Dict[str, Any]) -> Tuple[np.ndarray, float, float, float]:
    sc = normalize_scenario_schema(scenario)
    s = sc.get("scenario", {}) or {}
    legacy = sc.get("source", {}) or {}
    center = s.get("source_center") or s.get("impact_location") or s.get("load_area_center") or legacy.get("center", [0.0, 0.0, 0.0])
    center_np = np.asarray(_as_vec3(center), dtype=np.float32)
    radius = _as_float(s.get("source_radius", s.get("impact_radius", legacy.get("radius", 0.0))))
    t_src = _as_float(s.get("initial_temperature", legacy.get("initial_temperature", 0.0)))
    t_bg = _as_float(s.get("background_temperature", legacy.get("background_temperature", 0.0)))
    return center_np, radius, t_src, t_bg
