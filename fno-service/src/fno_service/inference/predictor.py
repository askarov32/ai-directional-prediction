from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..api.schemas import PredictionPayload
from ..data.dataset import FNOGridTensors, load_fno_grid_tensors
from ..models import FNO2d
from ..training.checkpoints import load_checkpoint
from ..utils.config import FNOServiceConfig


class CheckpointNotReadyError(RuntimeError):
    """Raised when FNO inference cannot run because no checkpoint is ready."""


class UnsupportedDomainError(ValueError):
    """Raised when the request cannot be represented by the current FNO baseline."""


class ModelLoadError(RuntimeError):
    """Raised when the FNO checkpoint or runtime artifacts cannot be loaded."""


class NonFiniteModelOutputError(RuntimeError):
    """Raised when the checkpoint produces NaN or infinite values."""


@dataclass(frozen=True)
class _RuntimeArtifacts:
    model: FNO2d
    tensors: FNOGridTensors
    checkpoint_path: Path
    model_version: str
    input_channels: list[str]
    output_channels: list[str]
    reference_temperature_k: float
    device: str


class FNOInferenceService:
    def __init__(self, config: FNOServiceConfig) -> None:
        self.config = config
        self._runtime: _RuntimeArtifacts | None = None

    def health_payload(self) -> dict[str, Any]:
        checkpoint = self._resolve_checkpoint_path()
        payload = {
            "status": "ok",
            "service": "fno-service",
            "ready": False,
            "mode": self._mode(),
            "checkpoint_path": str(self.config.checkpoint_path),
            "resolved_checkpoint_path": str(checkpoint) if checkpoint else None,
            "checkpoint_exists": bool(checkpoint and checkpoint.exists()),
            "dataset_path": str(self.config.dataset_path),
            "dataset_exists": self.config.dataset_path.exists(),
            "config_path": str(self.config.config_path),
            "device": self._resolved_device(),
            "allow_fallback": self.config.allow_fallback,
        }
        readiness = self._readiness_details()
        payload.update(readiness)
        return payload

    def readiness_payload(self) -> dict[str, Any]:
        payload = self.health_payload()
        payload["status"] = "ready" if payload["ready"] else "not_ready"
        return payload

    def predict(self, payload: PredictionPayload) -> dict[str, Any]:
        checkpoint = self._resolve_checkpoint_path()
        if checkpoint and checkpoint.exists():
            try:
                runtime = self._get_runtime()
            except Exception as exc:  # noqa: BLE001
                if self.config.allow_fallback:
                    response = self._fallback_prediction(payload)
                    response["diagnostics"]["checkpoint_runtime_error"] = str(exc)
                    response["diagnostics"]["mode"] = "fallback"
                    return response
                raise
            return self._checkpoint_prediction(payload, runtime)
        if self.config.allow_fallback:
            return self._fallback_prediction(payload)
        raise CheckpointNotReadyError(
            f"FNO checkpoint is not ready at {self.config.checkpoint_path}. "
            "Train a checkpoint or set FNO_ALLOW_FALLBACK=true for local skeleton wiring."
        )

    def _readiness_details(self) -> dict[str, Any]:
        checkpoint = self._resolve_checkpoint_path()
        if checkpoint and checkpoint.exists():
            try:
                runtime = self._get_runtime()
            except Exception as exc:  # noqa: BLE001
                if self.config.allow_fallback:
                    return {
                        "ready": True,
                        "mode": "fallback",
                        "checkpoint_loaded": False,
                        "reason": str(exc),
                        "fallback_reason": "checkpoint_runtime_not_usable",
                    }
                return {
                    "ready": False,
                    "mode": "not_ready",
                    "checkpoint_loaded": False,
                    "reason": str(exc),
                }
            return {
                "ready": True,
                "mode": "checkpoint",
                "checkpoint_loaded": True,
                "input_channels": runtime.input_channels,
                "output_channels": runtime.output_channels,
                "model_version": runtime.model_version,
            }
        if self.config.allow_fallback:
            return {
                "ready": True,
                "mode": "fallback",
                "checkpoint_loaded": False,
                "reason": "checkpoint missing, fallback mode enabled",
            }
        return {
            "ready": False,
            "mode": "not_ready",
            "checkpoint_loaded": False,
            "reason": f"checkpoint file not found at {self.config.checkpoint_path}",
        }

    def _mode(self) -> str:
        checkpoint = self._resolve_checkpoint_path()
        if checkpoint and checkpoint.exists():
            return "checkpoint"
        if self.config.allow_fallback:
            return "fallback"
        return "not_ready"

    def _resolve_checkpoint_path(self) -> Path | None:
        path = self.config.checkpoint_path
        if path.is_file():
            return path
        if path.is_dir():
            for name in ("best_model.pth", "model.pth"):
                candidate = path / name
                if candidate.exists():
                    return candidate
        return path

    def _resolved_device(self) -> str:
        requested = self.config.device.strip().lower()
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return requested or "cpu"

    def _get_runtime(self) -> _RuntimeArtifacts:
        if self._runtime is not None:
            return self._runtime

        checkpoint_path = self._resolve_checkpoint_path()
        if checkpoint_path is None or not checkpoint_path.exists():
            raise ModelLoadError(f"Checkpoint file does not exist: {self.config.checkpoint_path}")
        if not self.config.dataset_path.exists():
            raise ModelLoadError(f"Dataset directory does not exist: {self.config.dataset_path}")

        checkpoint = load_checkpoint(checkpoint_path, map_location=self._resolved_device())
        model_config = checkpoint.get("model_config")
        if not isinstance(model_config, dict):
            raise ModelLoadError("Checkpoint is missing model_config.")

        try:
            model = FNO2d(
                in_channels=int(model_config["in_channels"]),
                out_channels=int(model_config["out_channels"]),
                width=int(model_config["width"]),
                modes_x=int(model_config["modes_x"]),
                modes_y=int(model_config["modes_y"]),
                depth=int(model_config["depth"]),
            )
            model.load_state_dict(checkpoint["model_state_dict"])
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(f"Failed to load FNO checkpoint weights: {exc}") from exc

        model.to(torch.device(self._resolved_device()))
        model.eval()

        try:
            tensors = load_fno_grid_tensors(self.config.dataset_path)
        except FileNotFoundError as exc:
            raise ModelLoadError(str(exc)) from exc
        except ValueError as exc:
            raise ModelLoadError(f"Invalid FNO dataset at {self.config.dataset_path}: {exc}") from exc
        if tensors.grid_dynamic.shape[2] != 1:
            raise UnsupportedDomainError(
                "Current FNO2d inference supports only Z=1 regular grids. Prepare the dataset with a single Z slice."
            )

        channel_metadata = checkpoint.get("channel_metadata") or {}
        input_channels = _coerce_string_list(channel_metadata.get("input_channels"))
        output_channels = _coerce_string_list(channel_metadata.get("target_channels"))
        if not input_channels:
            input_channels = _default_input_channels(tensors)
        if not output_channels:
            output_channels = ["temperature_k", "disp_x", "disp_y", "disp_z"]

        reference_temperature_k = float(
            checkpoint.get("dataset_metadata", {}).get(
                "reference_temperature_k",
                tensors.metadata.get("reference_temperature_k", 293.15),
            )
        )
        runtime = _RuntimeArtifacts(
            model=model,
            tensors=tensors,
            checkpoint_path=checkpoint_path,
            model_version=f"fno-baseline@{checkpoint_path.name}",
            input_channels=input_channels,
            output_channels=output_channels,
            reference_temperature_k=reference_temperature_k,
            device=self._resolved_device(),
        )
        self._smoke_runtime(runtime)
        self._runtime = runtime
        return runtime

    def _smoke_runtime(self, runtime: _RuntimeArtifacts) -> None:
        sample = self._build_input_tensor(None, runtime)
        with torch.no_grad():
            outputs = runtime.model(sample.to(runtime.device))
        if outputs.ndim != 4 or outputs.shape[1] != len(runtime.output_channels):
            raise ModelLoadError(
                f"Unexpected FNO output shape {tuple(outputs.shape)} for output channels {runtime.output_channels}."
            )
        if not torch.isfinite(outputs).all():
            raise NonFiniteModelOutputError("FNO smoke inference produced NaN or infinite values.")

    def _checkpoint_prediction(self, payload: PredictionPayload, runtime: _RuntimeArtifacts) -> dict[str, Any]:
        _validate_request_domain(payload)
        inputs = self._build_input_tensor(payload, runtime)
        with torch.no_grad():
            outputs = runtime.model(inputs.to(runtime.device)).detach().cpu().numpy()[0]
        if not np.all(np.isfinite(outputs)):
            raise NonFiniteModelOutputError("FNO checkpoint inference produced NaN or infinite values.")

        source_cell = _resolve_grid_cell(runtime.tensors.grid_coords, payload.source, payload.domain)
        probe_cell = _resolve_grid_cell(runtime.tensors.grid_coords, payload.probe, payload.domain)
        direction = _prediction_direction(outputs, runtime.output_channels, source_cell=source_cell, probe_cell=probe_cell)

        azimuth = math.degrees(math.atan2(direction[1], direction[0]))
        horizontal = math.sqrt(direction[0] ** 2 + direction[1] ** 2)
        elevation = math.degrees(math.atan2(direction[2], horizontal))
        magnitude = _vector_norm(
            _sample_vector(outputs, runtime.output_channels, ("disp_x", "disp_y", "disp_z"), probe_cell)
        )
        dx = float(payload.probe.get("x", 0.0)) - float(payload.source.get("x", 0.0))
        dy = float(payload.probe.get("y", 0.0)) - float(payload.source.get("y", 0.0))
        dz = float(payload.probe.get("z", 0.0)) - float(payload.source.get("z", 0.0))
        distance = math.sqrt(max(dx * dx + dy * dy + dz * dz, 1e-8))
        vp = float(payload.medium.get("properties", {}).get("vp", 5.0) or 5.0)
        time_ms = float(payload.scenario.get("time_ms", 1.0))

        temperature_field = _channel_or_zeros(outputs, runtime.output_channels, "temperature_k")
        disp_x = _channel_or_zeros(outputs, runtime.output_channels, "disp_x")
        disp_y = _channel_or_zeros(outputs, runtime.output_channels, "disp_y")
        disp_z = _channel_or_zeros(outputs, runtime.output_channels, "disp_z")
        displacement_norm = np.sqrt(disp_x**2 + disp_y**2 + disp_z**2)

        return {
            "prediction": {
                "direction_vector": [round(value, 6) for value in direction],
                "azimuth_deg": round(azimuth, 4),
                "elevation_deg": round(elevation, 4),
                "magnitude": round(float(magnitude), 6),
                "wave_type": "fno_checkpoint_inference",
                "travel_time_ms": round(max((distance / max(vp, 0.1)) * 10.0 + time_ms * 0.05, 0.001), 6),
            },
            "field_summary": {
                "max_displacement": round(float(np.max(displacement_norm)), 8),
                "max_temperature_perturbation": round(
                    float(np.max(np.abs(temperature_field - runtime.reference_temperature_k))),
                    8,
                ),
            },
            "model_version": runtime.model_version,
            "diagnostics": {
                "checkpoint_loaded": True,
                "device": runtime.device,
                "input_channels": runtime.input_channels,
                "output_channels": runtime.output_channels,
                "source_cell": source_cell,
                "probe_cell": probe_cell,
                "mode": "checkpoint",
            },
        }

    def _build_input_tensor(self, payload: PredictionPayload | None, runtime: _RuntimeArtifacts) -> torch.Tensor:
        tensors = runtime.tensors
        field_names = tensors.field_names
        base_time_index = 0 if payload is None else _resolve_time_index(payload.scenario, tensors.metadata, tensors.grid_dynamic.shape[0])
        dynamic = np.array(tensors.grid_dynamic[base_time_index], copy=True)

        if payload is not None:
            source_cell = _resolve_grid_cell(tensors.grid_coords, payload.source, payload.domain)
            source_mask = _gaussian_mask(dynamic.shape[2:], source_cell)
            temperature_c = float(payload.scenario.get("temperature_c", 20.0))
            pressure_mpa = float(payload.scenario.get("pressure_mpa", 1.0))
            amplitude = float(payload.source.get("amplitude", 1.0))
            frequency_hz = float(payload.source.get("frequency_hz", 1.0))
            source_direction = _normalize(
                [float(value) for value in payload.source.get("direction", [1.0, 0.0, 0.0])]
            )

            if "temperature_k" in field_names:
                temp_index = field_names.index("temperature_k")
                base_temperature = runtime.reference_temperature_k + temperature_c
                dynamic[temp_index, 0] = dynamic[temp_index, 0] * 0.35 + base_temperature * 0.65
                dynamic[temp_index, 0] += source_mask * (0.15 * amplitude * max(frequency_hz, 1.0))
                dynamic[temp_index, 0] += pressure_mpa * 0.02

            for channel_name, direction_component in zip(("disp_x", "disp_y", "disp_z"), source_direction, strict=True):
                if channel_name in field_names:
                    channel_index = field_names.index(channel_name)
                    dynamic[channel_index, 0] += source_mask * direction_component * amplitude * 0.01

        input_chunks = [dynamic]
        if "youngs_modulus" in tensors.static_feature_names or tensors.grid_static.shape[0] > 0:
            input_chunks.append(tensors.grid_static)
        if tensors.grid_masks.shape[0] > 0:
            input_chunks.append(tensors.grid_masks)
        input_chunks.append(tensors.grid_coords)
        input_chunks.append(_time_channel(base_time_index, tensors.grid_dynamic.shape[0], dynamic.shape[1:]))

        inputs = np.concatenate(input_chunks, axis=0).astype(np.float32, copy=False)
        if inputs.shape[0] != len(runtime.input_channels):
            raise ModelLoadError(
                f"Input channel count mismatch. Built {inputs.shape[0]} channels but checkpoint expects {len(runtime.input_channels)}."
            )
        return torch.from_numpy(inputs[:, 0, :, :]).unsqueeze(0)

    def _fallback_prediction(self, payload: PredictionPayload) -> dict[str, Any]:
        source = payload.source
        probe = payload.probe
        scenario = payload.scenario
        medium = payload.medium
        properties = medium.get("properties", {}) if isinstance(medium, dict) else {}

        dx = float(probe.get("x", 0.0)) - float(source.get("x", 0.0))
        dy = float(probe.get("y", 0.0)) - float(source.get("y", 0.0))
        dz = float(probe.get("z", 0.0)) - float(source.get("z", 0.0))
        direction = _normalize([dx + 0.06, dy - 0.03, dz])

        azimuth = math.degrees(math.atan2(direction[1], direction[0]))
        horizontal = math.sqrt(direction[0] ** 2 + direction[1] ** 2)
        elevation = math.degrees(math.atan2(direction[2], horizontal))
        distance = math.sqrt(max(dx * dx + dy * dy + dz * dz, 1e-8))
        vp = float(properties.get("vp", 5.0) or 5.0)
        temperature = float(scenario.get("temperature_c", 20.0))
        pressure = float(scenario.get("pressure_mpa", 1.0))
        time_ms = float(scenario.get("time_ms", 1.0))

        return {
            "prediction": {
                "direction_vector": [round(value, 6) for value in direction],
                "azimuth_deg": round(azimuth, 4),
                "elevation_deg": round(elevation, 4),
                "magnitude": 1.0,
                "wave_type": "fno_skeleton_fallback",
                "travel_time_ms": round(max((distance / max(vp, 0.1)) * 10.0 + time_ms * 0.12, 0.001), 6),
            },
            "field_summary": {
                "max_displacement": round(0.001 + temperature / 350000.0 + pressure / 300000.0, 8),
                "max_temperature_perturbation": round(max(temperature, 0.0) / 160.0 + 0.45, 8),
            },
            "model_version": "fno-skeleton-fallback-v0",
            "diagnostics": {
                "mode": self._mode(),
                "service_config": asdict(self.config),
                "note": "Fallback response because no FNO checkpoint is configured.",
            },
        }


def _validate_request_domain(payload: PredictionPayload) -> None:
    domain_type = payload.domain.get("type")
    resolution = payload.domain.get("resolution", {})
    if domain_type != "rect_2d":
        raise UnsupportedDomainError(f"Current FNO2d inference supports only rect_2d domains, got {domain_type!r}.")
    if int(resolution.get("nz", 1)) != 1:
        raise UnsupportedDomainError(f"Current FNO2d inference requires nz=1, got {resolution.get('nz')!r}.")


def _resolve_time_index(scenario: dict[str, Any], metadata: dict[str, Any], total_timesteps: int) -> int:
    time_ms = float(scenario.get("time_ms", 0.0))
    time_end = float(metadata.get("time_end", max(total_timesteps - 1, 1)))
    time_start = float(metadata.get("time_start", 0.0))
    time_end = max(time_end, time_start + 1e-8)
    fraction = (time_ms - time_start) / (time_end - time_start)
    fraction = min(max(fraction, 0.0), 1.0)
    return min(int(round(fraction * max(total_timesteps - 2, 0))), max(total_timesteps - 2, 0))


def _time_channel(time_index: int, total_steps: int, spatial_shape: tuple[int, int, int]) -> np.ndarray:
    denominator = max(total_steps - 1, 1)
    value = np.float32(time_index / denominator)
    return np.full((1, *spatial_shape), value, dtype=np.float32)


def _default_input_channels(tensors: FNOGridTensors) -> list[str]:
    return list(tensors.field_names) + list(tensors.static_feature_names) + list(tensors.mask_names) + [
        "coord_x",
        "coord_y",
        "coord_z",
        "time_fraction",
    ]


def _coerce_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _resolve_grid_cell(grid_coords: np.ndarray, point: dict[str, Any], domain: dict[str, Any]) -> tuple[int, int]:
    coords_x = _normalize_grid_axis(grid_coords[0, 0])
    coords_y = _normalize_grid_axis(grid_coords[1, 0])
    size = domain.get("size", {})
    lx = max(float(size.get("lx", 1.0)), 1e-8)
    ly = max(float(size.get("ly", 1.0)), 1e-8)
    point_x = min(max(float(point.get("x", 0.0)) / lx, 0.0), 1.0)
    point_y = min(max(float(point.get("y", 0.0)) / ly, 0.0), 1.0)
    distance = (coords_x - point_x) ** 2 + (coords_y - point_y) ** 2
    y_index, x_index = np.unravel_index(int(np.argmin(distance)), distance.shape)
    return int(y_index), int(x_index)


def _normalize_grid_axis(values: np.ndarray) -> np.ndarray:
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if math.isclose(max_value, min_value):
        return np.zeros_like(values, dtype=np.float32)
    return ((values - min_value) / (max_value - min_value)).astype(np.float32, copy=False)


def _gaussian_mask(shape: tuple[int, int], center: tuple[int, int], sigma: float = 1.5) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.float32)
    center_y, center_x = center
    squared = (yy - center_y) ** 2 + (xx - center_x) ** 2
    return np.exp(-squared / (2.0 * sigma * sigma)).astype(np.float32, copy=False)


def _prediction_direction(
    outputs: np.ndarray,
    output_channels: list[str],
    *,
    source_cell: tuple[int, int],
    probe_cell: tuple[int, int],
) -> list[float]:
    sampled = _sample_vector(outputs, output_channels, ("disp_x", "disp_y", "disp_z"), probe_cell)
    geometric = [float(probe_cell[1] - source_cell[1]), float(probe_cell[0] - source_cell[0]), 0.0]
    return _normalize([sampled[0] + 0.25 * geometric[0], sampled[1] + 0.25 * geometric[1], sampled[2]])


def _sample_vector(
    outputs: np.ndarray,
    output_channels: list[str],
    component_names: tuple[str, str, str],
    cell: tuple[int, int],
) -> list[float]:
    y_index, x_index = cell
    return [
        float(_channel_or_zeros(outputs, output_channels, name)[y_index, x_index])
        for name in component_names
    ]


def _channel_or_zeros(outputs: np.ndarray, output_channels: list[str], name: str) -> np.ndarray:
    if name not in output_channels:
        return np.zeros(outputs.shape[1:], dtype=np.float32)
    return outputs[output_channels.index(name)]


def _normalize(vector: list[float]) -> list[float]:
    magnitude = _vector_norm(vector)
    if magnitude == 0:
        return [1.0, 0.0, 0.0]
    return [component / magnitude for component in vector]


def _vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(component * component for component in vector))
