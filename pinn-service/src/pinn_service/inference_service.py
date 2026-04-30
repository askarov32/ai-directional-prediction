from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pinn_service.inference_config import InferenceConfig
from pinn_service.inference_utils import build_feature_vector, build_prediction_payload
from pinn_service.model import MLP_PINN
from pinn_service.service_schemas import PINNPredictionRequest


class CheckpointNotReadyError(RuntimeError):
    """Raised when the PINN checkpoint is missing or failed to load."""


@dataclass(frozen=True)
class LoadedInferenceArtifacts:
    model: MLP_PINN
    device: torch.device
    input_feature_names: list[str]
    output_feature_names: list[str]
    input_mean: np.ndarray
    input_std: np.ndarray
    output_mean: np.ndarray
    output_std: np.ndarray
    best_loss: float
    checkpoint_path: Path


class PINNInferenceService:
    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.logger = logging.getLogger("pinn_service.inference")
        self._artifacts: LoadedInferenceArtifacts | None = None
        self._load_error: str | None = None

    def try_initialize(self) -> None:
        try:
            self._artifacts = self._load_artifacts(self.config.checkpoint_path, self.config.device)
            self._load_error = None
        except Exception as exc:  # noqa: BLE001
            self._artifacts = None
            self._load_error = str(exc)
            self.logger.warning("PINN checkpoint is not ready: %s", exc)

    def health_payload(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "pinn-service",
            "ready": self._artifacts is not None,
            "checkpoint_path": str(self.config.checkpoint_path),
            "device": self.config.device,
            "load_error": self._load_error,
            "model_version": self.model_version if self._artifacts is not None else None,
        }

    @property
    def model_version(self) -> str:
        if self._artifacts is None:
            return "unloaded"
        return f"pinn-baseline@{self._artifacts.checkpoint_path.name}"

    def predict(self, request: PINNPredictionRequest) -> dict[str, Any]:
        artifacts = self._require_artifacts()
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

        payload = build_prediction_payload(
            request=request,
            model_outputs=outputs,
            reference_temperature_k=self.config.reference_temperature_k,
        )
        payload["model_version"] = self.model_version
        return payload

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

        model = MLP_PINN(
            input_dim=len(input_feature_names),
            output_dim=len(output_feature_names),
            hidden_dim=int(config.get("hidden_dim", 192)),
            depth=int(config.get("depth", 6)),
            activation=str(config.get("activation", "tanh")),
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
