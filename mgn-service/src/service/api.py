from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MGN_DATASET_ID = os.getenv("MGN_DATASET_ID", "sandstone_comsol_real")
MGN_CONFIG_PATH = os.getenv("MGN_CONFIG_PATH", "configs/inference.yaml")
MGN_CHECKPOINT_PATH = os.getenv(
    "MGN_CHECKPOINT_PATH",
    "outputs/checkpoints_finetuned/best_model.pt",
)
MGN_DEVICE = os.getenv("MGN_DEVICE", "cuda")
MGN_TIMEOUT_SECONDS = int(os.getenv("MGN_PREDICT_TIMEOUT_SECONDS", "600"))
MGN_ROLLOUT_STEPS = os.getenv("MGN_ROLLOUT_STEPS", "5")
MGN_ALLOW_FALLBACK = os.getenv("MGN_ALLOW_FALLBACK", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
FIELD_GRID_DEFAULT_RESOLUTION = 64
FIELD_GRID_MAX_RESOLUTION = 128
FIELD_GRID_MISSING_FIELDS = [
    "stress_xx_pa",
    "stress_yy_pa",
    "stress_zz_pa",
    "stress_xy_pa",
    "stress_xz_pa",
    "stress_yz_pa",
    "stress_von_mises_pa",
    "strain_xx",
    "strain_yy",
    "strain_zz",
    "strain_xy",
    "strain_xz",
    "strain_yz",
    "velocity_x_m_s",
    "velocity_y_m_s",
    "velocity_z_m_s",
    "velocity_magnitude_m_s",
]


def resolve_checkpoint_path() -> Path:
    configured_path = PROJECT_ROOT / MGN_CHECKPOINT_PATH
    if configured_path.exists():
        return configured_path

    for candidate in (
        PROJECT_ROOT / "outputs" / "checkpoints" / "best_model.pt",
        PROJECT_ROOT / "outputs" / "checkpoints_finetuned" / "best_model.pt",
    ):
        if candidate.exists():
            return candidate

    return configured_path


class PredictionPayload(BaseModel):
    medium: dict[str, Any]
    scenario: dict[str, Any]
    source: dict[str, Any]
    probe: dict[str, Any]
    domain: dict[str, Any]
    representation: str = "directional"
    requested_outputs: list[str] = Field(default_factory=list)
    grid_policy: str | None = None


app = FastAPI(
    title="Real MeshGraphNet Service",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "meshgraphnet",
    }


@app.get("/ready")
async def ready() -> dict[str, Any]:
    dataset_dir = PROJECT_ROOT / "datasets" / MGN_DATASET_ID
    checkpoint_path = resolve_checkpoint_path()

    dataset_exists = dataset_dir.exists()
    checkpoint_exists = checkpoint_path.exists()
    real_ready = dataset_exists and checkpoint_exists
    ready_status = real_ready or MGN_ALLOW_FALLBACK

    return {
        "status": "ready" if ready_status else "not_ready",
        "ready": ready_status,
        "service": "meshgraphnet",
        "dataset_id": MGN_DATASET_ID,
        "dataset_dir": str(dataset_dir),
        "configured_checkpoint": str(PROJECT_ROOT / MGN_CHECKPOINT_PATH),
        "checkpoint": str(checkpoint_path),
        "checkpoint_exists": checkpoint_exists,
        "dataset_exists": dataset_exists,
        "device": MGN_DEVICE,
        "mode": "rollout" if real_ready else "fallback",
        "fallback_enabled": MGN_ALLOW_FALLBACK,
        "model_version": "real-meshgraphnet-v1" if real_ready else "mgn-service-fallback-v1",
    }


def normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))

    if norm == 0:
        return [1.0, 0.0, 0.0]

    return [x / norm for x in vector]


def make_direction_response(payload: PredictionPayload) -> dict[str, Any]:
    source = payload.source
    probe = payload.probe
    scenario = payload.scenario
    medium = payload.medium
    properties = medium.get("properties", {}) if isinstance(medium, dict) else {}

    dx = float(probe.get("x", 0.0)) - float(source.get("x", 0.0))
    dy = float(probe.get("y", 0.0)) - float(source.get("y", 0.0))
    dz = float(probe.get("z", 0.0)) - float(source.get("z", 0.0))

    direction = normalize_vector([dx, dy, dz])

    azimuth_deg = math.degrees(math.atan2(direction[1], direction[0]))
    horizontal = math.sqrt(direction[0] ** 2 + direction[1] ** 2)
    elevation_deg = math.degrees(math.atan2(direction[2], horizontal))
    distance = math.sqrt(max(dx * dx + dy * dy + dz * dz, 1e-8))
    vp = float(properties.get("vp", 5.0) or 5.0)
    temperature_c = float(scenario.get("temperature_c", 20.0))
    pressure_mpa = float(scenario.get("pressure_mpa", 1.0))
    time_ms = float(scenario.get("time_ms", 1.0))

    travel_time_ms = round(
        max((distance / max(vp, 0.1)) * 10.0 + time_ms * 0.15, 0.001), 6
    )
    max_displacement = round(
        0.0012 + temperature_c / 300000.0 + pressure_mpa / 250000.0, 8
    )
    max_temperature_perturbation = round(max(temperature_c, 0.0) / 140.0 + 0.5, 8)
    return {
        "direction_vector": [round(x, 6) for x in direction],
        "azimuth_deg": round(azimuth_deg, 4),
        "elevation_deg": round(elevation_deg, 4),
        "magnitude": 1.0,
        "wave_type": "dominant_p",
        "travel_time_ms": travel_time_ms,
        "max_displacement": max_displacement,
        "max_temperature_perturbation": max_temperature_perturbation,
        "model_version": "real-meshgraphnet-v1",
        # api-contract-v2 §7.1 — additive v2 blocks. MGN returns the
        # deterministic stub here, so per-point u/v aren't available.
        "schema_version": "2.0",
        "prediction_raw": {
            "temperature_k": None,
            "temperature_perturbation_k": max_temperature_perturbation,
            "displacement_m": {"u": None, "v": None},
            "travel_time_s": travel_time_ms / 1000.0,
            "response_magnitude_score": 1.0,
        },
        "optional_outputs": {
            "confidence_score": None,
            "field_summary": {
                "max_temperature_k": None,
                "max_displacement_m": max_displacement,
                "max_temperature_perturbation_k": max_temperature_perturbation,
                "max_von_mises_stress_pa": None,
            },
            "field_grid": None,
            "field_sources": {},
            "available_fields": [],
            "missing_fields": (
                list(FIELD_GRID_MISSING_FIELDS)
                if _should_emit_field_grid(payload)
                else []
            ),
            "strain": None,
            "stress": None,
        },
        "diagnostics": {
            "fallback_used": False,
            "fallback_reason": None,
            "warnings": [],
            "mode": "deterministic",
        },
    }


def _get_metric_value(
    summary: dict[str, Any],
    metric_name: str,
    value_key: str = "max_final",
) -> float | None:
    metric = summary.get(metric_name)

    if not isinstance(metric, dict):
        return None

    value = metric.get(value_key)

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def enrich_response_from_summary(response: dict[str, Any]) -> dict[str, Any]:
    summary_path = PROJECT_ROOT / "outputs" / "predictions" / "summary_metrics.json"

    if not summary_path.exists():
        return response

    try:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception:
        return response

    max_displacement = _get_metric_value(
        summary,
        "displacement_magnitude",
        "max_final",
    )

    max_temperature_perturbation = _get_metric_value(
        summary,
        "temperature_change",
        "max_final",
    )

    max_von_mises_stress = _get_metric_value(
        summary,
        "von_mises_stress",
        "max_final",
    )

    max_velocity = _get_metric_value(
        summary,
        "velocity_magnitude",
        "max_final",
    )

    risk_flag = _get_metric_value(
        summary,
        "risk_flag",
        "max_final",
    )

    if max_displacement is not None:
        response["max_displacement"] = max_displacement

    if max_temperature_perturbation is not None:
        response["max_temperature_perturbation"] = max_temperature_perturbation

    # Эти поля можно использовать позже, если backend/frontend расширить.
    response["extra_metrics"] = {
        "max_von_mises_stress": max_von_mises_stress,
        "max_velocity": max_velocity,
        "risk_flag": risk_flag,
    }

    response["summary_metrics_path"] = str(summary_path)

    # api-contract-v2 §7.1 — propagate the enriched (real-rollout)
    # values into the v2 blocks that make_direction_response() seeded
    # with the deterministic stub values.
    opt = response.get("optional_outputs") or {}
    fs = opt.setdefault("field_summary", {})
    if max_displacement is not None:
        fs["max_displacement_m"] = max_displacement
    if max_temperature_perturbation is not None:
        fs["max_temperature_perturbation_k"] = max_temperature_perturbation
    if max_von_mises_stress is not None:
        fs["max_von_mises_stress_pa"] = max_von_mises_stress
    response["optional_outputs"] = opt

    raw = response.get("prediction_raw") or {}
    if max_temperature_perturbation is not None:
        raw["temperature_perturbation_k"] = max_temperature_perturbation
    response["prediction_raw"] = raw

    diag = response.get("diagnostics") or {}
    diag["mode"] = "rollout"
    diag["rollout_source"] = "summary_metrics.json"
    response["diagnostics"] = diag

    return response


def _should_emit_field_grid(payload: PredictionPayload) -> bool:
    requested = {
        str(item).strip().lower()
        for item in payload.requested_outputs
        if str(item).strip()
    }
    return "field_grid" in requested or "all" in requested


def _load_prediction_bundle() -> dict[str, Any] | None:
    prediction_path = PROJECT_ROOT / "outputs" / "predictions" / "prediction.pt"
    if not prediction_path.exists():
        return None
    try:
        import torch as _torch

        bundle = _torch.load(prediction_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    return bundle if isinstance(bundle, dict) else None


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)
    except Exception:
        return None


def _field_index(field_names: list[str], candidates: list[str]) -> int | None:
    lower = [field.strip().lower() for field in field_names]
    for candidate in candidates:
        cand = candidate.strip().lower()
        for index, field in enumerate(lower):
            if field == cand or field.endswith("." + cand) or field.endswith("_" + cand):
                return index
    return None


def _derived_final_values(
    derived: dict[str, Any],
    name: str,
    node_count: int,
) -> np.ndarray | None:
    values = _to_numpy(derived.get(name))
    if values is None:
        return None
    if values.ndim == 2 and values.shape[1] == node_count:
        return values[-1].astype(np.float32, copy=False)
    if values.ndim == 1 and values.shape[0] == node_count:
        return values.astype(np.float32, copy=False)
    return None


def _field_grid_resolution(payload: PredictionPayload) -> tuple[int, int]:
    policy = (payload.grid_policy or "service_default").strip().lower()
    cap = (
        FIELD_GRID_MAX_RESOLUTION
        if policy in {"high", "full", "native", "128", "max"}
        else FIELD_GRID_DEFAULT_RESOLUTION
    )
    resolution = (
        payload.domain.get("resolution", {})
        if isinstance(payload.domain, dict)
        else {}
    )
    nx = min(max(int(resolution.get("nx", cap)), 2), cap)
    ny = min(max(int(resolution.get("ny", cap)), 2), cap)
    return nx, ny


def _coords_to_domain(values: np.ndarray, scale: float) -> np.ndarray:
    coords = np.asarray(values, dtype=np.float32)
    min_value = float(np.min(coords))
    max_value = float(np.max(coords))
    if math.isclose(min_value, max_value):
        return np.zeros_like(coords, dtype=np.float32)
    normalized = (coords - min_value) / (max_value - min_value)
    return normalized.astype(np.float32, copy=False) * float(scale)


def _nearest_node_indices(coords_xy: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    indices: list[np.ndarray] = []
    chunk_size = 128
    for start in range(0, points_xy.shape[0], chunk_size):
        chunk = points_xy[start : start + chunk_size]
        delta = chunk[:, None, :] - coords_xy[None, :, :]
        distances = np.sum(delta * delta, axis=2)
        indices.append(np.argmin(distances, axis=1).astype(np.int64, copy=False))
    return np.concatenate(indices, axis=0)


def _sample_to_grid(
    values: np.ndarray,
    nearest: np.ndarray,
    *,
    ny: int,
    nx: int,
) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)[nearest].reshape(ny, nx)


def _field_channel(
    *,
    label: str,
    group: str,
    unit: str,
    values: np.ndarray,
    source: str,
    decimals: int,
) -> dict[str, Any]:
    return {
        "label": label,
        "group": group,
        "unit": unit,
        "values": np.round(
            np.asarray(values, dtype=np.float64),
            decimals=decimals,
        ).tolist(),
        "source": source,
    }


def _append_warning(response: dict[str, Any], warning: str) -> dict[str, Any]:
    diagnostics = response.get("diagnostics") or {}
    warnings = list(diagnostics.get("warnings") or [])
    if warning not in warnings:
        warnings.append(warning)
    diagnostics["warnings"] = warnings
    response["diagnostics"] = diagnostics
    return response


def enrich_response_with_field_grid(
    response: dict[str, Any],
    payload: PredictionPayload,
) -> dict[str, Any]:
    if not _should_emit_field_grid(payload):
        return response
    if payload.domain.get("type") != "rect_2d":
        return _append_warning(response, "field_grid_not_available_for_domain")

    bundle = _load_prediction_bundle()
    if bundle is None:
        return _append_warning(response, "field_grid_not_returned_by_model")

    trajectory = _to_numpy(bundle.get("trajectory"))
    coords = _to_numpy(bundle.get("coords"))
    field_names = bundle.get("field_names")
    if (
        trajectory is None
        or coords is None
        or field_names is None
        or trajectory.ndim != 3
        or coords.ndim != 2
        or coords.shape[1] < 2
        or trajectory.shape[0] == 0
    ):
        return _append_warning(response, "field_grid_not_returned_by_model")

    field_list = [str(name) for name in field_names]
    final_step = trajectory[-1].astype(np.float32, copy=False)
    node_count = final_step.shape[0]
    if (
        node_count == 0
        or coords.shape[0] != node_count
        or final_step.shape[1] != len(field_list)
    ):
        return _append_warning(response, "field_grid_not_returned_by_model")

    size = payload.domain.get("size", {}) if isinstance(payload.domain, dict) else {}
    lx = float(size.get("lx", 1.0) or 1.0)
    ly = float(size.get("ly", 1.0) or 1.0)
    nx, ny = _field_grid_resolution(payload)
    x_coords = np.linspace(0.0, lx, nx, dtype=np.float32)
    y_coords = np.linspace(0.0, ly, ny, dtype=np.float32)
    mesh_x, mesh_y = np.meshgrid(x_coords, y_coords)
    points_xy = np.column_stack([mesh_x.ravel(), mesh_y.ravel()]).astype(np.float32)
    coords_xy = np.column_stack(
        [
            _coords_to_domain(coords[:, 0], lx),
            _coords_to_domain(coords[:, 1], ly),
        ]
    ).astype(np.float32, copy=False)
    nearest = _nearest_node_indices(coords_xy, points_xy)

    temperature_idx = _field_index(
        field_list,
        ["t", "temp", "temperature", "temperature_k"],
    )
    disp_x_idx = _field_index(field_list, ["u", "disp_x", "displacement_x"])
    disp_y_idx = _field_index(field_list, ["v", "disp_y", "displacement_y"])
    disp_z_idx = _field_index(field_list, ["w", "disp_z", "displacement_z"])
    zeros = np.zeros(node_count, dtype=np.float32)

    temperature_values = (
        final_step[:, temperature_idx] if temperature_idx is not None else zeros
    )
    disp_x_values = final_step[:, disp_x_idx] if disp_x_idx is not None else zeros
    disp_y_values = final_step[:, disp_y_idx] if disp_y_idx is not None else zeros
    disp_z_values = zeros
    if disp_z_idx is not None and payload.domain.get("type") != "rect_2d":
        disp_z_values = final_step[:, disp_z_idx]

    derived = bundle.get("derived") if isinstance(bundle.get("derived"), dict) else {}
    temp_delta_values = _derived_final_values(derived, "temperature_change", node_count)
    if temp_delta_values is None:
        temp_delta_values = temperature_values - 293.15
    displacement_values = _derived_final_values(
        derived,
        "displacement_magnitude",
        node_count,
    )
    if displacement_values is None:
        displacement_values = np.sqrt(
            disp_x_values**2 + disp_y_values**2 + disp_z_values**2
        )

    temperature = _sample_to_grid(temperature_values, nearest, ny=ny, nx=nx)
    temperature_delta = _sample_to_grid(temp_delta_values, nearest, ny=ny, nx=nx)
    disp_x = _sample_to_grid(disp_x_values, nearest, ny=ny, nx=nx)
    disp_y = _sample_to_grid(disp_y_values, nearest, ny=ny, nx=nx)
    disp_z = _sample_to_grid(disp_z_values, nearest, ny=ny, nx=nx)
    displacement_magnitude = _sample_to_grid(
        displacement_values,
        nearest,
        ny=ny,
        nx=nx,
    )

    channels = {
        "temperature_k": _field_channel(
            label="Temperature",
            group="temperature",
            unit="K",
            values=temperature,
            source="direct_model_output" if temperature_idx is not None else "not_available",
            decimals=6,
        ),
        "temperature_perturbation_k": _field_channel(
            label="Temperature perturbation",
            group="temperature",
            unit="K",
            values=temperature_delta,
            source="derived_from_temperature",
            decimals=6,
        ),
        "disp_x_m": _field_channel(
            label="Displacement X",
            group="displacement",
            unit="m",
            values=disp_x,
            source="direct_model_output" if disp_x_idx is not None else "not_available",
            decimals=12,
        ),
        "disp_y_m": _field_channel(
            label="Displacement Y",
            group="displacement",
            unit="m",
            values=disp_y,
            source="direct_model_output" if disp_y_idx is not None else "not_available",
            decimals=12,
        ),
        "displacement_magnitude_m": _field_channel(
            label="Displacement magnitude",
            group="displacement",
            unit="m",
            values=displacement_magnitude,
            source="derived_from_displacement_components",
            decimals=12,
        ),
    }

    opt = response.get("optional_outputs") or {}
    field_summary = opt.get("field_summary") or {}
    field_summary.update(
        {
            "max_temperature_k": round(float(np.max(temperature)), 6),
            "max_temperature_perturbation_k": round(
                float(np.max(np.abs(temperature_delta))),
                8,
            ),
            "max_displacement_m": round(float(np.max(displacement_magnitude)), 8),
            "max_von_mises_stress_pa": field_summary.get("max_von_mises_stress_pa"),
        }
    )
    opt.update(
        {
            "field_summary": field_summary,
            "field_grid": {
                "type": "rect_2d",
                "nx": nx,
                "ny": ny,
                "x_coords": np.round(x_coords.astype(np.float64), decimals=8).tolist(),
                "y_coords": np.round(y_coords.astype(np.float64), decimals=8).tolist(),
                "channels": channels,
            },
            "field_sources": {
                name: channel["source"] for name, channel in channels.items()
            },
            "available_fields": [
                name
                for name, channel in channels.items()
                if channel["source"] != "not_available"
            ],
            "missing_fields": list(FIELD_GRID_MISSING_FIELDS),
        }
    )
    response["optional_outputs"] = opt
    diagnostics = response.get("diagnostics") or {}
    diagnostics["field_grid"] = {
        "emitted": True,
        "type": "rect_2d",
        "nx": nx,
        "ny": ny,
        "point_count": nx * ny,
        "source": "prediction.pt",
        "method": "nearest_node_resampling",
        "policy": payload.grid_policy or "service_default",
    }
    response["diagnostics"] = diagnostics
    return response


def enrich_response_with_probe_sample(
    response: dict[str, Any], probe: dict[str, Any]
) -> dict[str, Any]:
    """Sample the trained MGN trajectory at the probe node so the v2
    prediction_raw block carries real per-point T, u, v values
    (not just the field-aggregate summary)."""
    bundle = _load_prediction_bundle()
    if bundle is None:
        return response
    traj = bundle.get("trajectory")
    coords = bundle.get("coords")
    field_names = bundle.get("field_names")
    if traj is None or coords is None or field_names is None:
        return response
    traj_np = _to_numpy(traj)
    coords_np = _to_numpy(coords)
    if (
        traj_np is None
        or coords_np is None
        or traj_np.ndim != 3
        or coords_np.ndim != 2
        or coords_np.shape[0] == 0
        or coords_np.shape[1] < 2
        or traj_np.shape[0] == 0
        or traj_np.shape[1] != coords_np.shape[0]
        or traj_np.shape[2] != len(field_names)
    ):
        return response
    field_list = [str(name) for name in field_names]
    px = float(probe.get("x", 0.0))
    py = float(probe.get("y", 0.0))
    pz = float(probe.get("z", 0.0))
    # nearest-node index on coords[:, :3]
    probe_coords = np.zeros((coords_np.shape[0], 3), dtype=coords_np.dtype)
    probe_coords[:, : min(coords_np.shape[1], 3)] = coords_np[
        :,
        : min(coords_np.shape[1], 3),
    ]
    diffs = probe_coords - np.array([px, py, pz], dtype=coords_np.dtype)
    node_idx = int(np.argmin(np.sum(diffs * diffs, axis=1)))
    last_step = traj_np[-1, node_idx, :]  # (F,)
    temperature_idx = _field_index(
        field_list,
        ["t", "temp", "temperature", "temperature_k"],
    )
    disp_x_idx = _field_index(field_list, ["u", "disp_x", "displacement_x"])
    disp_y_idx = _field_index(field_list, ["v", "disp_y", "displacement_y"])

    raw = response.get("prediction_raw") or {}
    if temperature_idx is not None:
        raw["temperature_k"] = float(last_step[temperature_idx])
    if disp_x_idx is not None:
        raw.setdefault("displacement_m", {})
        raw["displacement_m"]["u"] = float(last_step[disp_x_idx])
    if disp_y_idx is not None:
        raw.setdefault("displacement_m", {})
        raw["displacement_m"]["v"] = float(last_step[disp_y_idx])
    response["prediction_raw"] = raw
    diag = response.get("diagnostics") or {}
    diag["probe_node_index"] = node_idx
    response["diagnostics"] = diag
    return response


@app.post("/predict")
async def predict(payload: PredictionPayload) -> dict[str, Any]:
    dataset_dir = PROJECT_ROOT / "datasets" / MGN_DATASET_ID
    checkpoint_path = resolve_checkpoint_path()

    if not dataset_dir.exists():
        if MGN_ALLOW_FALLBACK:
            response = make_direction_response(payload)
            response["model_version"] = "mgn-service-fallback-v1"
            response["extra_metrics"] = {
                "fallback_reason": "dataset_not_found",
                "dataset_id": MGN_DATASET_ID,
                "expected_path": str(dataset_dir),
            }
            return response
        raise HTTPException(
            status_code=503,
            detail={
                "message": "MeshGraphNet dataset not found",
                "dataset_id": MGN_DATASET_ID,
                "expected_path": str(dataset_dir),
            },
        )

    if not checkpoint_path.exists():
        if MGN_ALLOW_FALLBACK:
            response = make_direction_response(payload)
            response["model_version"] = "mgn-service-fallback-v1"
            response["extra_metrics"] = {
                "fallback_reason": "checkpoint_not_found",
                "configured_path": str(PROJECT_ROOT / MGN_CHECKPOINT_PATH),
                "expected_path": str(checkpoint_path),
            }
            return response
        raise HTTPException(
            status_code=503,
            detail={
                "message": "MeshGraphNet checkpoint not found",
                "expected_path": str(checkpoint_path),
            },
        )

    cmd = [
        sys.executable,
        "scripts/run_prediction.py",
        "--config",
        MGN_CONFIG_PATH,
        "--dataset_id",
        MGN_DATASET_ID,
        "--checkpoint",
        str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "--rollout_steps",
        MGN_ROLLOUT_STEPS,
        "--no_animate",
        "--no_vtk",
        "--no_plots",
    ]

    summary_path = PROJECT_ROOT / "outputs" / "predictions" / "summary_metrics.json"
    _safe_unlink(summary_path)

    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=MGN_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail={
                "message": "MeshGraphNet prediction timeout",
                "timeout_seconds": MGN_TIMEOUT_SECONDS,
            },
        )

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "MeshGraphNet prediction script failed",
                "command": cmd,
                "stdout_tail": result.stdout[-4000:],
                "stderr_tail": result.stderr[-4000:],
            },
        )

    response = make_direction_response(payload)
    response = enrich_response_from_summary(response)
    response = enrich_response_with_probe_sample(
        response, payload.probe if hasattr(payload, "probe") else {}
    )
    response = enrich_response_with_field_grid(response, payload)

    return response
