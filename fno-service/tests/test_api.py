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
    assert payload["prediction"]["direction_vector"][2] == 0.0
    assert payload["prediction"]["elevation_deg"] == 0.0
    assert payload["diagnostics"]["normalization_used"] is True
    assert payload["diagnostics"]["denormalization_used"] is True
    assert payload["diagnostics"]["effective_domain_type"] == "rect_2d"
    assert payload["diagnostics"]["domain_adaptation"] == "none"
    assert payload["model_version"].startswith("fno-baseline@")


def test_checkpoint_predict_returns_requested_field_grid(tmp_path: Path) -> None:
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
    payload = sample_payload()
    payload["requested_outputs"] = [
        "field_grid",
        "field_summary",
        "probe_sample",
        "diagnostics",
    ]

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    field_grid = body["optional_outputs"]["field_grid"]
    assert field_grid["type"] == "rect_2d"
    assert field_grid["nx"] == 6
    assert field_grid["ny"] == 6
    assert set(field_grid["channels"]) >= {
        "temperature_k",
        "temperature_perturbation_k",
        "disp_x_m",
        "disp_y_m",
        "disp_z_m",
        "displacement_magnitude_m",
    }
    assert field_grid["channels"]["temperature_k"]["group"] == "temperature"
    assert field_grid["channels"]["disp_x_m"]["source"] == "direct_model_output"
    assert field_grid["channels"]["disp_z_m"]["source"] == "derived_from_2d_domain"
    assert len(field_grid["channels"]["temperature_k"]["values"]) == 6
    assert len(field_grid["channels"]["temperature_k"]["values"][0]) == 6
    assert "stress_von_mises_pa" in body["optional_outputs"]["missing_fields"]
    assert body["optional_outputs"]["field_sources"][
        "displacement_magnitude_m"
    ] == "derived_from_displacement_components"


def test_predict_returns_400_for_unsupported_domain(tmp_path: Path) -> None:
    dataset_dir = write_tiny_2d_fno_dataset(tmp_path / "dataset")
    checkpoint_path = train_tiny_fno_checkpoint(dataset_dir, tmp_path / "checkpoint")
    config = FNOServiceConfig(
        checkpoint_path=checkpoint_path,
        config_path=tmp_path / "inference.yaml",
        dataset_path=dataset_dir,
        device="cuda",
        log_level="INFO",
        service_port=9000,
        allow_fallback=False,
    )
    client = TestClient(create_app(config))
    payload = sample_payload()
    payload["domain"]["type"] = "rect_3d"
    payload["domain"]["size"]["lz"] = 1.0
    payload["domain"]["resolution"]["nz"] = 8

    response = client.post("/predict", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "UNSUPPORTED_DOMAIN"
