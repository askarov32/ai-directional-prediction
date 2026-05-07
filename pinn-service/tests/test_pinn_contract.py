from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from pinn_service.inference_config import InferenceConfig
from pinn_service.inference_service import CheckpointNotReadyError, PINNInferenceService
from pinn_service.service_schemas import PINNPredictionRequest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "pinn-service" / "artifacts" / "checkpoints" / "baseline_quick"


def sample_payload() -> dict:
    return {
        "medium": {
            "id": "sandstone_medium",
            "name": "Sandstone (medium)",
            "category": "sedimentary",
            "properties": {
                "rho": 2684.0,
                "porosity_total": 0.34,
                "porosity_effective": 0.27,
                "vp": 6.17,
                "vs": 3.2,
                "thermal_conductivity": 2.5,
                "heat_capacity": 850.0,
                "thermal_expansion": 0.000012,
            },
            "ranges": {"temperature_c": [-20.0, 300.0], "pressure_mpa": [0.1, 1500.0]},
            "metadata": {"source": "test"},
        },
        "scenario": {"temperature_c": 120.0, "pressure_mpa": 35.0, "time_ms": 12.0},
        "source": {
            "type": "thermal_pulse",
            "x": 0.15,
            "y": 0.4,
            "z": 0.0,
            "amplitude": 1.0,
            "frequency_hz": 50.0,
            "direction": [1.0, 0.0, 0.0],
        },
        "probe": {"x": 0.7, "y": 0.55, "z": 0.0},
        "domain": {
            "type": "rect_2d",
            "size": {"lx": 1.0, "ly": 1.0, "lz": 0.0},
            "resolution": {"nx": 128, "ny": 128, "nz": 1},
            "boundary_conditions": {
                "left": "fixed",
                "right": "free",
                "top": "insulated",
                "bottom": "insulated",
            },
        },
        "representation": "physics_informed",
        "routing_hint": "pinn",
    }


def inference_config(checkpoint_path: Path) -> InferenceConfig:
    return InferenceConfig(
        checkpoint_path=checkpoint_path,
        device="cpu",
        log_level="INFO",
        service_port=9000,
        reference_temperature_k=293.15,
        time_scale=1.0,
    )


def test_pinn_request_rejects_extra_fields():
    payload = sample_payload()
    payload["unexpected"] = "nope"

    with pytest.raises(ValidationError):
        PINNPredictionRequest.model_validate(payload)


def test_pinn_request_rejects_wrong_representation():
    payload = sample_payload()
    payload["representation"] = "grid"

    with pytest.raises(ValidationError):
        PINNPredictionRequest.model_validate(payload)


def test_missing_checkpoint_is_not_ready(tmp_path):
    service = PINNInferenceService(inference_config(tmp_path / "missing.pth"))

    service.try_initialize()

    ready = service.readiness_payload()
    assert ready["ready"] is False
    assert ready["status"] == "not_ready"
    with pytest.raises(CheckpointNotReadyError):
        service.predict(PINNPredictionRequest.model_validate(sample_payload()))


def test_checkpoint_smoke_check_exposes_metadata_and_diagnostics():
    service = PINNInferenceService(inference_config(CHECKPOINT_DIR))

    service.try_initialize()

    ready = service.readiness_payload()
    assert ready["ready"] is True
    assert ready["smoke_check"]["status"] == "passed"
    assert ready["active_feature_count"] == len(ready["active_feature_names"])

    response = service.predict(PINNPredictionRequest.model_validate(sample_payload()))
    assert "model_outputs" in response
    assert "postprocessed_prediction" in response
    assert "diagnostics" in response
    assert response["diagnostics"]["smoke_check"]["status"] == "passed"
