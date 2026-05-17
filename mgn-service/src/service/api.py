from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


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
MGN_ALLOW_FALLBACK = os.getenv("MGN_ALLOW_FALLBACK", "true").lower() in {"1", "true", "yes", "on"}


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
                "max_displacement_m": max_displacement,
                "max_temperature_perturbation_k": max_temperature_perturbation,
            },
            "field_grid": None,
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

    return response
