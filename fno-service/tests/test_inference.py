from __future__ import annotations

from pathlib import Path

import torch

from fno_service.api.schemas import PredictionPayload
from fno_service.inference.predictor import FNOInferenceService, _sanity_warnings
from fno_service.utils.config import FNOServiceConfig

from .helpers import train_tiny_fno_checkpoint, write_tiny_2d_fno_dataset


def sample_payload() -> PredictionPayload:
    return PredictionPayload.model_validate(
        {
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
    )


def test_checkpoint_inference_returns_backend_compatible_payload(tmp_path: Path) -> None:
    dataset_dir = write_tiny_2d_fno_dataset(tmp_path / "dataset")
    checkpoint_path = train_tiny_fno_checkpoint(dataset_dir, tmp_path / "checkpoint")
    service = FNOInferenceService(
        FNOServiceConfig(
            checkpoint_path=checkpoint_path,
            config_path=tmp_path / "inference.yaml",
            dataset_path=dataset_dir,
            device="cpu",
            log_level="INFO",
            service_port=9000,
            allow_fallback=False,
        )
    )

    payload = service.predict(sample_payload())

    assert payload["model_version"].startswith("fno-baseline@")
    assert len(payload["prediction"]["direction_vector"]) == 3
    assert payload["prediction"]["wave_type"] == "fno_checkpoint_inference"
    assert payload["prediction"]["direction_vector"][2] == 0.0
    assert payload["prediction"]["elevation_deg"] == 0.0
    assert 0.0 <= payload["prediction"]["magnitude"] <= 1.0
    assert payload["field_summary"]["max_displacement"] >= 0.0
    assert payload["diagnostics"]["checkpoint_loaded"] is True
    assert payload["diagnostics"]["effective_domain_type"] == "rect_2d"
    assert payload["diagnostics"]["domain_adaptation"] == "none"
    assert payload["diagnostics"]["fallback_used"] is False
    assert payload["diagnostics"]["normalization_used"] is True
    assert payload["diagnostics"]["denormalization_used"] is True
    assert payload["diagnostics"]["normalization"]["mode"] == "channel_wise_standardization"


def test_checkpoint_inference_falls_back_to_cpu_when_cuda_unavailable(tmp_path: Path, monkeypatch) -> None:
    dataset_dir = write_tiny_2d_fno_dataset(tmp_path / "dataset")
    checkpoint_path = train_tiny_fno_checkpoint(dataset_dir, tmp_path / "checkpoint")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    service = FNOInferenceService(
        FNOServiceConfig(
            checkpoint_path=checkpoint_path,
            config_path=tmp_path / "inference.yaml",
            dataset_path=dataset_dir,
            device="cuda",
            log_level="INFO",
            service_port=9000,
            allow_fallback=False,
        )
    )

    readiness = service.readiness_payload()
    payload = service.predict(sample_payload())

    assert readiness["ready"] is True
    assert readiness["device"] == "cpu"
    assert payload["diagnostics"]["device"] == "cpu"


def test_checkpoint_inference_rejects_nonzero_2d_z_values(tmp_path: Path) -> None:
    dataset_dir = write_tiny_2d_fno_dataset(tmp_path / "dataset")
    checkpoint_path = train_tiny_fno_checkpoint(dataset_dir, tmp_path / "checkpoint")
    service = FNOInferenceService(
        FNOServiceConfig(
            checkpoint_path=checkpoint_path,
            config_path=tmp_path / "inference.yaml",
            dataset_path=dataset_dir,
            device="cpu",
            log_level="INFO",
            service_port=9000,
            allow_fallback=False,
        )
    )
    payload = sample_payload()
    payload.source["z"] = 0.1

    try:
        service.predict(payload)
    except Exception as exc:  # noqa: BLE001
        assert "source.z=0.0" in str(exc)
    else:
        raise AssertionError("Expected FNO rect_2d inference to reject nonzero source.z.")


def test_sanity_warnings_flag_missing_normalization_and_scale_outliers() -> None:
    warnings = _sanity_warnings(
        max_displacement=101.0,
        max_temperature_perturbation=10001.0,
        magnitude=1.0,
        normalization_used=False,
        denormalization_used=False,
    )

    assert "missing_input_normalization_metadata" in warnings
    assert "missing_output_denormalization_metadata" in warnings
    assert "scale_outlier:max_displacement_gt_1e2" in warnings
    assert "scale_outlier:max_temperature_perturbation_gt_1e4" in warnings
