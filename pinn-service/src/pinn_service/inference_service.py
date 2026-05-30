from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from pinn_service.inference_config import InferenceConfig
from pinn_service.inference_utils import build_feature_vector, build_prediction_payload
from pinn_service.model import create_pinn_model, parse_layer_dims
from pinn_service.service_schemas import PINNPredictionRequest


class CheckpointNotReadyError(RuntimeError):
    """Raised when the PINN checkpoint is missing or failed to load."""


_FIELD_GRID_DEFAULT_RESOLUTION = 64
_FIELD_GRID_MAX_RESOLUTION = 128
_FIELD_GRID_CHUNK_SIZE = 4096
_FIELD_GRID_MISSING_FIELDS = [
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


@dataclass(frozen=True)
class LoadedInferenceArtifacts:
    model: nn.Module
    device: torch.device
    input_feature_names: list[str]
    output_feature_names: list[str]
    input_mean: np.ndarray
    input_std: np.ndarray
    output_mean: np.ndarray
    output_std: np.ndarray
    best_loss: float
    checkpoint_path: Path
    architecture: str


class PINNInferenceService:
    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.logger = logging.getLogger("pinn_service.inference")
        self._artifacts: LoadedInferenceArtifacts | None = None
        self._load_error: str | None = None
        self._smoke_check: dict[str, Any] = {"status": "not_run"}

    def try_initialize(self) -> None:
        try:
            artifacts = self._load_artifacts(self.config.checkpoint_path, self.config.device)
            self._run_smoke_check(artifacts)
            self._artifacts = artifacts
            self._load_error = None
        except Exception as exc:  # noqa: BLE001
            self._artifacts = None
            self._load_error = str(exc)
            if self._smoke_check.get("status") != "failed":
                self._smoke_check = {"status": "not_run"}
            self.logger.warning("PINN checkpoint is not ready: %s", exc)

    def health_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": "ok",
            "service": "pinn-service",
            "ready": self._artifacts is not None,
            "checkpoint_path": str(self.config.checkpoint_path),
            "device": self.config.device,
            "load_error": self._load_error,
            "model_version": self.model_version if self._artifacts is not None else None,
            "smoke_check": self._smoke_check,
        }
        if self._artifacts is not None:
            payload.update(self._checkpoint_metadata(self._artifacts))
        return payload

    def readiness_payload(self) -> dict[str, Any]:
        payload = self.health_payload()
        ready = bool(payload["ready"])
        payload["status"] = "ready" if ready else "not_ready"
        return payload

    @property
    def model_version(self) -> str:
        if self._artifacts is None:
            return "unloaded"
        return f"pinn-{self._artifacts.architecture}@{self._artifacts.checkpoint_path.name}"

    def predict(self, request: PINNPredictionRequest) -> dict[str, Any]:
        artifacts = self._require_artifacts()
        features = build_feature_vector(
            request,
            time_scale=self.config.time_scale,
            expected_feature_names=artifacts.input_feature_names,
        )
        self._assert_feature_alignment(features.feature_names, artifacts.input_feature_names)

        outputs = self._predict_feature_matrix(features.values[None, :], artifacts, chunk_size=1)[0]

        payload = build_prediction_payload(
            request=request,
            model_outputs=outputs,
            reference_temperature_k=self.config.reference_temperature_k,
        )
        payload["model_version"] = self.model_version
        payload["model_outputs"] = {
            "feature_names": artifacts.output_feature_names,
            "values": [round(float(value), 9) for value in outputs.tolist()],
        }
        payload["postprocessed_prediction"] = {
            key: payload[key]
            for key in (
                "direction_vector",
                "azimuth_deg",
                "elevation_deg",
                "magnitude",
                "wave_type",
                "travel_time_ms",
                "max_displacement",
                "max_temperature_perturbation",
            )
        }
        payload["diagnostics"] = {
            **self._checkpoint_metadata(artifacts),
            "smoke_check": self._smoke_check,
            "postprocessing_note": (
                "Flat response fields combine neural outputs with geometry/material postprocessing "
                "to keep the MVP API backward-compatible."
            ),
            # api-contract-v2 §7.1 — fallback flag always present.
            "fallback_used": False,
            "fallback_reason": None,
            "warnings": [],
        }

        # api-contract-v2 §7.1 — additive v2 blocks shipped alongside v1
        # flat fields. The backend's remote_response_schema_v2 prefers
        # these when present so v2 callers get direct per-point fields.
        feature_names = artifacts.output_feature_names or []
        output_map = (
            dict(zip(feature_names, [float(value) for value in outputs.tolist()]))
            if feature_names
            else {}
        )
        field_grid_payload = (
            self._build_field_grid(request, artifacts)
            if _should_emit_field_grid(request)
            else None
        )
        max_temperature_k = (
            field_grid_payload["field_summary"]["max_temperature_k"]
            if field_grid_payload
            else output_map.get("temperature_k")
        )
        max_temperature_perturbation_k = (
            field_grid_payload["field_summary"]["max_temperature_perturbation_k"]
            if field_grid_payload
            else payload["max_temperature_perturbation"]
        )
        max_displacement_m = (
            field_grid_payload["field_summary"]["max_displacement_m"]
            if field_grid_payload
            else payload["max_displacement"]
        )

        payload["schema_version"] = "2.0"
        payload["prediction_raw"] = {
            "temperature_k": output_map.get("temperature_k"),
            "temperature_perturbation_k": (
                output_map["temperature_k"] - self.config.reference_temperature_k
                if "temperature_k" in output_map
                else None
            ),
            "displacement_m": {
                "u": output_map.get("disp_x"),
                "v": output_map.get("disp_y"),
            },
            "travel_time_s": payload["travel_time_ms"] / 1000.0,
            "response_magnitude_score": payload.get("magnitude"),
        }
        payload["optional_outputs"] = {
            "confidence_score": None,
            "field_summary": {
                "max_temperature_k": max_temperature_k,
                "max_temperature_perturbation_k": max_temperature_perturbation_k,
                "max_displacement_m": max_displacement_m,
                "max_von_mises_stress_pa": None,
            },
            "field_grid": field_grid_payload["field_grid"] if field_grid_payload else None,
            "field_sources": field_grid_payload["field_sources"] if field_grid_payload else {},
            "available_fields": (
                field_grid_payload["available_fields"] if field_grid_payload else []
            ),
            "missing_fields": field_grid_payload["missing_fields"] if field_grid_payload else [],
            "strain": None,
            "stress": None,
        }
        if field_grid_payload:
            payload["diagnostics"]["field_grid"] = field_grid_payload["diagnostics"]
        elif _should_emit_field_grid(request):
            payload["diagnostics"]["warnings"].append(
                "field_grid_not_available_for_domain"
                if request.domain.type != "rect_2d"
                else "field_grid_not_returned_by_model"
            )
        return payload

    def _predict_feature_matrix(
        self,
        feature_matrix: np.ndarray,
        artifacts: LoadedInferenceArtifacts,
        *,
        chunk_size: int,
    ) -> np.ndarray:
        matrix = np.asarray(feature_matrix, dtype=np.float32)
        outputs: list[np.ndarray] = []

        artifacts.model.eval()
        with torch.no_grad():
            for start in range(0, matrix.shape[0], chunk_size):
                chunk = matrix[start : start + chunk_size]
                scaled = ((chunk - artifacts.input_mean) / artifacts.input_std).astype(np.float32)
                tensor = torch.from_numpy(scaled).to(artifacts.device)
                outputs.append(artifacts.model(tensor).cpu().numpy())

        output_matrix = np.concatenate(outputs, axis=0)
        denormalized = output_matrix * artifacts.output_std + artifacts.output_mean
        if not np.all(np.isfinite(denormalized)):
            raise CheckpointNotReadyError("PINN inference produced NaN or infinite values.")
        return denormalized

    def _build_field_grid(
        self,
        request: PINNPredictionRequest,
        artifacts: LoadedInferenceArtifacts,
    ) -> dict[str, Any] | None:
        if request.domain.type != "rect_2d":
            return None

        nx, ny = _field_grid_resolution(request)
        lx = float(request.domain.size.lx)
        ly = float(request.domain.size.ly)
        x_coords = np.linspace(0.0, lx, nx, dtype=np.float32)
        y_coords = np.linspace(0.0, ly, ny, dtype=np.float32)
        feature_rows: list[np.ndarray] = []

        for y_coord in y_coords:
            for x_coord in x_coords:
                probe = request.probe.model_copy(
                    update={
                        "x": float(x_coord),
                        "y": float(y_coord),
                        "z": 0.0,
                    }
                )
                grid_request = request.model_copy(update={"probe": probe})
                features = build_feature_vector(
                    grid_request,
                    time_scale=self.config.time_scale,
                    expected_feature_names=artifacts.input_feature_names,
                )
                feature_rows.append(features.values)

        feature_matrix = np.stack(feature_rows).astype(np.float32, copy=False)
        outputs = self._predict_feature_matrix(
            feature_matrix,
            artifacts,
            chunk_size=_FIELD_GRID_CHUNK_SIZE,
        ).reshape(ny, nx, len(artifacts.output_feature_names))

        output_channels = artifacts.output_feature_names
        temperature = _output_grid_or_zeros(outputs, output_channels, "temperature_k")
        disp_x = _output_grid_or_zeros(outputs, output_channels, "disp_x")
        disp_y = _output_grid_or_zeros(outputs, output_channels, "disp_y")
        disp_z_available = "disp_z" in output_channels
        disp_z = np.zeros_like(disp_x, dtype=np.float32)
        if disp_z_available and request.domain.type != "rect_2d":
            disp_z = _output_grid_or_zeros(outputs, output_channels, "disp_z")

        temperature_perturbation = temperature - self.config.reference_temperature_k
        displacement_magnitude = np.sqrt(disp_x**2 + disp_y**2 + disp_z**2)

        channels = {
            "temperature_k": _field_channel(
                label="Temperature",
                group="temperature",
                unit="K",
                values=temperature,
                source="direct_model_output",
                decimals=6,
            ),
            "temperature_perturbation_k": _field_channel(
                label="Temperature perturbation",
                group="temperature",
                unit="K",
                values=temperature_perturbation,
                source="derived_from_temperature",
                decimals=6,
            ),
            "disp_x_m": _field_channel(
                label="Displacement X",
                group="displacement",
                unit="m",
                values=disp_x,
                source="direct_model_output",
                decimals=12,
            ),
            "disp_y_m": _field_channel(
                label="Displacement Y",
                group="displacement",
                unit="m",
                values=disp_y,
                source="direct_model_output",
                decimals=12,
            ),
            "disp_z_m": _field_channel(
                label="Displacement Z",
                group="displacement",
                unit="m",
                values=disp_z,
                source=(
                    "direct_model_output"
                    if disp_z_available and request.domain.type != "rect_2d"
                    else "derived_from_2d_domain"
                ),
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
        field_summary = {
            "max_temperature_k": round(float(np.max(temperature)), 6),
            "max_temperature_perturbation_k": round(
                float(np.max(np.abs(temperature_perturbation))),
                8,
            ),
            "max_displacement_m": round(float(np.max(displacement_magnitude)), 8),
            "max_von_mises_stress_pa": None,
        }
        return {
            "field_grid": {
                "type": "rect_2d",
                "nx": nx,
                "ny": ny,
                "x_coords": np.round(x_coords.astype(np.float64), decimals=8).tolist(),
                "y_coords": np.round(y_coords.astype(np.float64), decimals=8).tolist(),
                "channels": channels,
            },
            "field_summary": field_summary,
            "field_sources": {
                name: channel["source"] for name, channel in channels.items()
            },
            "available_fields": list(channels.keys()),
            "missing_fields": list(_FIELD_GRID_MISSING_FIELDS),
            "diagnostics": {
                "emitted": True,
                "type": "rect_2d",
                "nx": nx,
                "ny": ny,
                "point_count": nx * ny,
                "chunk_size": _FIELD_GRID_CHUNK_SIZE,
                "policy": request.grid_policy or "service_default",
                "method": "batched_probe_inference",
            },
        }

    def _require_artifacts(self) -> LoadedInferenceArtifacts:
        if self._artifacts is None:
            raise CheckpointNotReadyError(self._load_error or "PINN checkpoint is not loaded.")
        return self._artifacts

    def _load_artifacts(self, checkpoint_path: Path, device_name: str) -> LoadedInferenceArtifacts:
        resolved_checkpoint = self._resolve_checkpoint_path(checkpoint_path)
        if not resolved_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint}")

        device = torch.device(device_name)
        checkpoint = torch.load(resolved_checkpoint, map_location=device)
        config = checkpoint.get("config", {})
        input_feature_names = checkpoint["input_feature_names"]
        output_feature_names = checkpoint["output_feature_names"]

        model = create_pinn_model(
            input_dim=len(input_feature_names),
            output_dim=len(output_feature_names),
            architecture=str(config.get("architecture", "mlp")),
            hidden_dim=int(config.get("hidden_dim", 192)),
            depth=int(config.get("depth", 6)),
            activation=str(config.get("activation", "tanh")),
            mlp_layer_dims=_read_mlp_layer_dims(config.get("mlp_layer_dims")),
            num_blocks=int(config.get("num_blocks", 4)),
            use_fourier_features=bool(config.get("use_fourier_features", False)),
            fourier_num_frequencies=int(config.get("fourier_num_frequencies", 6)),
            fourier_scale=float(config.get("fourier_scale", 1.0)),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        input_scaler = checkpoint["input_scaler"]
        output_scaler = checkpoint["output_scaler"]

        return LoadedInferenceArtifacts(
            model=model,
            device=device,
            input_feature_names=list(input_feature_names),
            output_feature_names=list(output_feature_names),
            input_mean=np.asarray(input_scaler["mean"], dtype=np.float32),
            input_std=np.asarray(input_scaler["std"], dtype=np.float32),
            output_mean=np.asarray(output_scaler["mean"], dtype=np.float32),
            output_std=np.asarray(output_scaler["std"], dtype=np.float32),
            best_loss=float(checkpoint.get("best_loss", 0.0)),
            checkpoint_path=resolved_checkpoint,
            architecture=str(config.get("architecture", "mlp")),
        )

    def _run_smoke_check(self, artifacts: LoadedInferenceArtifacts) -> None:
        request = self._build_smoke_request()
        features = build_feature_vector(
            request,
            time_scale=self.config.time_scale,
            expected_feature_names=artifacts.input_feature_names,
        )
        self._assert_feature_alignment(features.feature_names, artifacts.input_feature_names)

        scaled = ((features.values - artifacts.input_mean) / artifacts.input_std).astype(np.float32)
        tensor = torch.from_numpy(scaled[None, :]).to(artifacts.device)

        artifacts.model.eval()
        with torch.no_grad():
            outputs_scaled = artifacts.model(tensor).cpu().numpy()[0]
        outputs = outputs_scaled * artifacts.output_std + artifacts.output_mean

        if outputs.shape[0] != len(artifacts.output_feature_names):
            self._smoke_check = {
                "status": "failed",
                "reason": "unexpected_output_shape",
                "expected": len(artifacts.output_feature_names),
                "actual": int(outputs.shape[0]),
            }
            raise CheckpointNotReadyError("PINN smoke check failed: unexpected output shape.")
        if outputs.shape[0] < 4 or not np.all(np.isfinite(outputs)):
            self._smoke_check = {
                "status": "failed",
                "reason": "non_finite_or_insufficient_outputs",
                "output_feature_count": int(outputs.shape[0]),
            }
            raise CheckpointNotReadyError("PINN smoke check failed: outputs are invalid.")

        payload = build_prediction_payload(
            request=request,
            model_outputs=outputs,
            reference_temperature_k=self.config.reference_temperature_k,
        )
        numeric_values: list[float] = [
            float(payload["azimuth_deg"]),
            float(payload["elevation_deg"]),
            float(payload["magnitude"]),
            float(payload["travel_time_ms"]),
            float(payload["max_displacement"]),
            float(payload["max_temperature_perturbation"]),
            *[float(component) for component in payload["direction_vector"]],
        ]
        if not all(np.isfinite(value) for value in numeric_values):
            self._smoke_check = {"status": "failed", "reason": "non_finite_postprocessed_payload"}
            raise CheckpointNotReadyError("PINN smoke check failed: postprocessed payload is invalid.")

        self._smoke_check = {
            "status": "passed",
            "output_feature_count": len(artifacts.output_feature_names),
        }

    def _checkpoint_metadata(self, artifacts: LoadedInferenceArtifacts) -> dict[str, Any]:
        return {
            "resolved_checkpoint_path": str(artifacts.checkpoint_path),
            "active_feature_count": len(artifacts.input_feature_names),
            "active_feature_names": artifacts.input_feature_names,
            "output_feature_count": len(artifacts.output_feature_names),
            "output_feature_names": artifacts.output_feature_names,
            "best_loss": artifacts.best_loss,
            "device": str(artifacts.device),
            "architecture": artifacts.architecture,
        }

    def _build_smoke_request(self) -> PINNPredictionRequest:
        return PINNPredictionRequest.model_validate(
            {
                "medium": {
                    "id": "smoke_sandstone",
                    "name": "Smoke Sandstone",
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
        )

    def _assert_feature_alignment(self, request_feature_names: list[str], checkpoint_feature_names: list[str]) -> None:
        if request_feature_names != checkpoint_feature_names:
            raise CheckpointNotReadyError(
                "Feature mismatch between inference request builder and checkpoint metadata."
            )

    def _resolve_checkpoint_path(self, checkpoint_path: Path) -> Path:
        if checkpoint_path.is_file():
            return checkpoint_path
        if checkpoint_path.is_dir():
            candidates = [
                checkpoint_path / "best_model.pth",
                checkpoint_path / "model.pth",
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            discovered = sorted(checkpoint_path.glob("*.pth"))
            if discovered:
                return discovered[0]
        return checkpoint_path


def _read_mlp_layer_dims(raw_value: Any) -> tuple[int, ...] | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        return parse_layer_dims(raw_value)
    if isinstance(raw_value, (list, tuple)):
        return tuple(int(value) for value in raw_value)
    raise ValueError(f"Unsupported mlp_layer_dims checkpoint value: {raw_value!r}")


def _should_emit_field_grid(request: PINNPredictionRequest) -> bool:
    requested = {
        str(item).strip().lower()
        for item in request.requested_outputs
        if str(item).strip()
    }
    return "field_grid" in requested or "all" in requested


def _field_grid_resolution(request: PINNPredictionRequest) -> tuple[int, int]:
    policy = (request.grid_policy or "service_default").strip().lower()
    cap = (
        _FIELD_GRID_MAX_RESOLUTION
        if policy in {"high", "full", "native", "128", "max"}
        else _FIELD_GRID_DEFAULT_RESOLUTION
    )
    nx = min(max(int(request.domain.resolution.nx), 2), cap)
    ny = min(max(int(request.domain.resolution.ny), 2), cap)
    return nx, ny


def _output_grid_or_zeros(
    outputs: np.ndarray,
    output_channels: list[str],
    name: str,
) -> np.ndarray:
    if name not in output_channels:
        return np.zeros(outputs.shape[:2], dtype=np.float32)
    return outputs[:, :, output_channels.index(name)].astype(np.float32, copy=False)


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
        "values": _matrix_values(values, decimals=decimals),
        "source": source,
    }


def _matrix_values(values: np.ndarray, *, decimals: int) -> list[list[float]]:
    matrix = np.asarray(values, dtype=np.float64)
    return np.round(matrix, decimals=decimals).tolist()
