from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from transformer_service.dataset import INPUT_CHANNEL_NAMES, TARGET_CHANNEL_NAMES
from transformer_service.inference_config import InferenceConfig
from transformer_service.inference_utils import (
    build_initial_state,
    build_prediction_payload,
)
from transformer_service.model import OFormer
from transformer_service.service_schemas import TransformerPredictionRequest
from transformer_service.tokenizer import (
    NormalizationStats,
    denormalize_target,
    normalize_state,
    update_state_with_prediction,
)


class CheckpointNotReadyError(RuntimeError):
    """Raised when the Transformer checkpoint is missing or failed to load."""


@dataclass(frozen=True)
class LoadedInferenceArtifacts:
    model: OFormer
    device: torch.device
    input_channel_names: list[str]
    target_channel_names: list[str]
    coords: np.ndarray
    stats: NormalizationStats
    best_loss: float
    checkpoint_path: Path
    config: dict[str, Any]


class TransformerInferenceService:
    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.logger = logging.getLogger("transformer_service.inference")
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
            self.logger.warning("Transformer checkpoint is not ready: %s", exc)

    def health_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": "ok",
            "service": "transformer-service",
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
        payload["status"] = "ready" if payload["ready"] else "not_ready"
        return payload

    @property
    def model_version(self) -> str:
        if self._artifacts is None:
            return "unloaded"
        return f"oformer-baseline@{self._artifacts.checkpoint_path.name}"

    def predict(self, request: TransformerPredictionRequest) -> dict[str, Any]:
        artifacts = self._require_artifacts()
        trajectory_raw = self._rollout(request, artifacts)
        payload = build_prediction_payload(
            request=request,
            trajectory_raw=trajectory_raw,
            reference_temperature_k=self.config.reference_temperature_k,
        )
        payload["model_version"] = self.model_version
        payload["model_outputs"] = {
            "feature_names": artifacts.target_channel_names,
            "final_step_values": [
                round(float(v), 9) for v in trajectory_raw[:, -1, :].mean(axis=0).tolist()
            ],
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
            "rollout_steps": int(trajectory_raw.shape[1]),
            "postprocessing_note": (
                "Baseline transformer-service inference uses checkpoint coords and "
                "autoregressive rollout; direction blends model output with source/probe geometry."
            ),
        }
        return payload

    def _rollout(
        self,
        request: TransformerPredictionRequest,
        artifacts: LoadedInferenceArtifacts,
    ) -> np.ndarray:
        initial = build_initial_state(
            request=request,
            coords=artifacts.coords,
            reference_temperature_k=self.config.reference_temperature_k,
        )
        state_norm = normalize_state(initial.state, artifacts.stats)
        coords_norm = (
            (initial.coords - artifacts.stats.input_mean[:3]) / artifacts.stats.input_std[:3]
        ).astype(np.float32)
        coords_t = torch.from_numpy(coords_norm).to(artifacts.device).unsqueeze(0)

        steps = max(int(self.config.rollout_steps), 1)
        traj_raw_steps: list[np.ndarray] = []
        artifacts.model.eval()
        with torch.no_grad():
            for _ in range(steps):
                tokens = torch.from_numpy(state_norm).to(artifacts.device).unsqueeze(0)
                pred_norm = artifacts.model(tokens, coords_t).cpu().numpy()[0]
                pred_raw = denormalize_target(pred_norm, artifacts.stats)
                traj_raw_steps.append(pred_raw)
                state_norm = update_state_with_prediction(state_norm, pred_raw, artifacts.stats)
        trajectory = np.stack(traj_raw_steps, axis=1)
        return trajectory

    def _require_artifacts(self) -> LoadedInferenceArtifacts:
        if self._artifacts is None:
            raise CheckpointNotReadyError(self._load_error or "Transformer checkpoint is not loaded.")
        return self._artifacts

    def _load_artifacts(self, checkpoint_path: Path, device_name: str) -> LoadedInferenceArtifacts:
        resolved_checkpoint = self._resolve_checkpoint_path(checkpoint_path)
        if not resolved_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint}")

        device = torch.device(device_name)
        checkpoint = torch.load(resolved_checkpoint, map_location=device, weights_only=False)
        config = checkpoint.get("config", {})
        input_channel_names = checkpoint["input_channel_names"]
        target_channel_names = checkpoint["target_channel_names"]
        if list(input_channel_names) != list(INPUT_CHANNEL_NAMES):
            raise CheckpointNotReadyError("Checkpoint input channels disagree with current schema.")
        if list(target_channel_names) != list(TARGET_CHANNEL_NAMES):
            raise CheckpointNotReadyError("Checkpoint target channels disagree with current schema.")

        model = OFormer(
            input_dim=len(input_channel_names),
            query_dim=3,
            output_dim=len(target_channel_names),
            d_model=int(config.get("d_model", 128)),
            n_heads=int(config.get("n_heads", 4)),
            enc_depth=int(config.get("enc_depth", 4)),
            dec_depth=int(config.get("dec_depth", 4)),
            ffn_expansion=int(config.get("ffn_expansion", 4)),
            dropout=float(config.get("dropout", 0.1)),
            activation=str(config.get("activation", "gelu")),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        stats = NormalizationStats(
            input_mean=np.asarray(checkpoint["input_mean"], dtype=np.float32),
            input_std=np.asarray(checkpoint["input_std"], dtype=np.float32),
            target_mean=np.asarray(checkpoint["target_mean"], dtype=np.float32),
            target_std=np.asarray(checkpoint["target_std"], dtype=np.float32),
        )
        coords = np.asarray(checkpoint["coords"], dtype=np.float32)

        return LoadedInferenceArtifacts(
            model=model,
            device=device,
            input_channel_names=list(input_channel_names),
            target_channel_names=list(target_channel_names),
            coords=coords,
            stats=stats,
            best_loss=float(checkpoint.get("best_loss", 0.0)),
            checkpoint_path=resolved_checkpoint,
            config=dict(config),
        )

    def _run_smoke_check(self, artifacts: LoadedInferenceArtifacts) -> None:
        request = self._build_smoke_request()
        trajectory = self._rollout_with_artifacts(request, artifacts, max_steps=3)
        if trajectory.shape[-1] != len(artifacts.target_channel_names):
            self._smoke_check = {
                "status": "failed",
                "reason": "unexpected_output_shape",
                "expected": len(artifacts.target_channel_names),
                "actual": int(trajectory.shape[-1]),
            }
            raise CheckpointNotReadyError("Transformer smoke check failed: shape mismatch.")
        if not np.all(np.isfinite(trajectory)):
            self._smoke_check = {"status": "failed", "reason": "non_finite_outputs"}
            raise CheckpointNotReadyError("Transformer smoke check failed: non-finite outputs.")
        payload = build_prediction_payload(
            request=request,
            trajectory_raw=trajectory,
            reference_temperature_k=self.config.reference_temperature_k,
        )
        for key in (
            "azimuth_deg",
            "elevation_deg",
            "magnitude",
            "travel_time_ms",
            "max_displacement",
            "max_temperature_perturbation",
        ):
            if not np.isfinite(float(payload[key])):
                self._smoke_check = {"status": "failed", "reason": "non_finite_postprocessed_payload"}
                raise CheckpointNotReadyError("Transformer smoke check failed: payload not finite.")
        if not all(np.isfinite(c) for c in payload["direction_vector"]):
            self._smoke_check = {"status": "failed", "reason": "non_finite_direction_vector"}
            raise CheckpointNotReadyError("Transformer smoke check failed: direction not finite.")
        self._smoke_check = {
            "status": "passed",
            "output_feature_count": len(artifacts.target_channel_names),
            "rollout_steps": int(trajectory.shape[1]),
        }

    def _rollout_with_artifacts(
        self,
        request: TransformerPredictionRequest,
        artifacts: LoadedInferenceArtifacts,
        max_steps: int,
    ) -> np.ndarray:
        saved_steps = self.config.rollout_steps
        try:
            object.__setattr__(self.config, "rollout_steps", max_steps)
            return self._rollout(request, artifacts)
        finally:
            object.__setattr__(self.config, "rollout_steps", saved_steps)

    def _checkpoint_metadata(self, artifacts: LoadedInferenceArtifacts) -> dict[str, Any]:
        return {
            "resolved_checkpoint_path": str(artifacts.checkpoint_path),
            "active_feature_count": len(artifacts.input_channel_names),
            "active_feature_names": artifacts.input_channel_names,
            "output_feature_count": len(artifacts.target_channel_names),
            "output_feature_names": artifacts.target_channel_names,
            "best_loss": artifacts.best_loss,
            "device": str(artifacts.device),
            "trained_node_count": int(artifacts.coords.shape[0]),
            "architecture": "oformer",
        }

    def _build_smoke_request(self) -> TransformerPredictionRequest:
        return TransformerPredictionRequest.model_validate(
            {
                "medium": {
                    "id": "smoke_sandstone",
                    "name": "Smoke Sandstone",
                    "category": "sedimentary",
                    "properties": {
                        "rho": 2200.0,
                        "porosity_total": 0.34,
                        "porosity_effective": 0.27,
                        "vp": 4.0,
                        "vs": 2.3,
                        "thermal_conductivity": 2.2,
                        "heat_capacity": 800.0,
                        "thermal_expansion": 1.0e-5,
                    },
                },
                "scenario": {"temperature_c": 20.0, "pressure_mpa": 5.0, "time_ms": 1.0},
                "source": {
                    "type": "thermal_pulse",
                    "x": 0.1,
                    "y": 0.1,
                    "z": 0.0,
                    "amplitude": 1.0,
                    "frequency_hz": 50.0,
                    "direction": [1.0, 0.0, 0.0],
                },
                "probe": {"x": 0.5, "y": 0.5, "z": 0.0},
                "domain": {
                    "type": "rect_2d",
                    "size": {"lx": 1.0, "ly": 1.0, "lz": 0.0},
                    "resolution": {"nx": 64, "ny": 64, "nz": 1},
                    "boundary_conditions": {
                        "left": "fixed",
                        "right": "free",
                        "top": "insulated",
                        "bottom": "insulated",
                    },
                },
                "representation": "tokenset",
                "routing_hint": "transformer",
            }
        )

    def _resolve_checkpoint_path(self, checkpoint_path: Path) -> Path:
        if checkpoint_path.is_file():
            return checkpoint_path
        if checkpoint_path.is_dir():
            for candidate_name in ("best_model.pth", "model.pth"):
                candidate = checkpoint_path / candidate_name
                if candidate.exists():
                    return candidate
            discovered = sorted(checkpoint_path.glob("*.pth"))
            if discovered:
                return discovered[0]
        return checkpoint_path
