from __future__ import annotations

from pathlib import Path

from fno_service.api.schemas import PredictionPayload
from fno_service.inference.predictor import FNOInferenceService
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
    assert payload["field_summary"]["max_displacement"] >= 0.0
    assert payload["diagnostics"]["checkpoint_loaded"] is True
