from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from fno_service.api.main import create_app
from fno_service.utils.config import FNOServiceConfig

from .helpers import train_tiny_fno_checkpoint, write_tiny_2d_fno_dataset


def make_client(tmp_path: Path, *, allow_fallback: bool = False) -> TestClient:
    config = FNOServiceConfig(
        checkpoint_path=tmp_path / "missing_checkpoint",
        config_path=tmp_path / "inference.yaml",
        dataset_path=tmp_path / "dataset",
        device="cpu",
        log_level="INFO",
        service_port=9000,
        allow_fallback=allow_fallback,
    )
    return TestClient(create_app(config))


def sample_payload() -> dict:
    return {
        "medium": {
            "id": "granite",
            "name": "Granite",
            "category": "igneous",
            "properties": {"vp": 5.95},
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
        "representation": "grid",
        "routing_hint": "fno",
    }


def test_health_does_not_require_checkpoint(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "fno-service"
    assert payload["ready"] is False


def test_ready_returns_503_when_checkpoint_missing(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_predict_returns_503_when_checkpoint_missing(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.post("/predict", json=sample_payload())

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "CHECKPOINT_NOT_READY"


def test_fallback_predict_returns_backend_compatible_payload(tmp_path: Path) -> None:
    client = make_client(tmp_path, allow_fallback=True)

    ready = client.get("/ready")
    response = client.post("/predict", json=sample_payload())

    assert ready.status_code == 200
    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"]["wave_type"] == "fno_skeleton_fallback"
    assert payload["field_summary"]["max_displacement"] > 0
    assert payload["model_version"] == "fno-skeleton-fallback-v0"


def test_checkpoint_predict_returns_real_inference_payload(tmp_path: Path) -> None:
    dataset_dir = write_tiny_2d_fno_dataset(tmp_path / "dataset")
    checkpoint_path = train_tiny_fno_checkpoint(dataset_dir, tmp_path / "checkpoint")
    config = FNOServiceConfig(
        checkpoint_path=checkpoint_path,
        config_path=tmp_path / "inference.yaml",
        dataset_path=dataset_dir,
        device="cpu",
        log_level="INFO",
        service_port=9000,
        allow_fallback=False,
    )
    client = TestClient(create_app(config))

    ready = client.get("/ready")
    response = client.post("/predict", json=sample_payload())

    assert ready.status_code == 200
    assert ready.json()["mode"] == "checkpoint"
    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"]["wave_type"] == "fno_checkpoint_inference"
    assert payload["model_version"].startswith("fno-baseline@")
