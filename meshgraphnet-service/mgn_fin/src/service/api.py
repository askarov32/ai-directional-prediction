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
    checkpoint_path = PROJECT_ROOT / MGN_CHECKPOINT_PATH

    dataset_exists = dataset_dir.exists()
    checkpoint_exists = checkpoint_path.exists()

    return {
        "status": "ready" if dataset_exists and checkpoint_exists else "not_ready",
        "ready": dataset_exists and checkpoint_exists,
        "service": "meshgraphnet",
        "dataset_id": MGN_DATASET_ID,
        "dataset_dir": str(dataset_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_exists": checkpoint_exists,
        "dataset_exists": dataset_exists,
        "device": MGN_DEVICE,
        "model_version": "real-meshgraphnet-v1",
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

    dx = float(probe.get("x", 0.0)) - float(source.get("x", 0.0))
    dy = float(probe.get("y", 0.0)) - float(source.get("y", 0.0))
    dz = float(probe.get("z", 0.0)) - float(source.get("z", 0.0))

    direction = normalize_vector([dx, dy, dz])

    azimuth_deg = math.degrees(math.atan2(direction[1], direction[0]))
    horizontal = math.sqrt(direction[0] ** 2 + direction[1] ** 2)
    elevation_deg = math.degrees(math.atan2(direction[2], horizontal))

    return {
        "direction_vector": [round(x, 6) for x in direction],
        "azimuth_deg": round(azimuth_deg, 4),
        "elevation_deg": round(elevation_deg, 4),
        "magnitude": 1.0,
        "wave_type": "dominant_p",
        "travel_time_ms": float(scenario.get("time_ms", 1.0)),
        "max_displacement": None,
        "max_temperature_perturbation": None,
        "model_version": "real-meshgraphnet-v1",
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
    checkpoint_path = PROJECT_ROOT / MGN_CHECKPOINT_PATH

    if not dataset_dir.exists():
        raise HTTPException(
            status_code=503,
            detail={
                "message": "MeshGraphNet dataset not found",
                "dataset_id": MGN_DATASET_ID,
                "expected_path": str(dataset_dir),
            },
        )

    if not checkpoint_path.exists():
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
        MGN_CHECKPOINT_PATH,
        "--rollout_steps",
        MGN_ROLLOUT_STEPS,
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