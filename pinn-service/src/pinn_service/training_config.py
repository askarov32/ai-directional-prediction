from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class TrainingConfig:
    dataset_path: Path
    output_dir: Path
    val_dataset_path: Path | None = None
    device: str = "cpu"
    epochs: int = 25
    batch_size: int = 4096
    validation_batch_size: int | None = None
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-6
    hidden_dim: int = 192
    depth: int = 6
    activation: str = "tanh"
    supervised_weight: float = 1.0
    velocity_weight: float = 0.25
    wave_residual_weight: float = 0.1
    thermal_residual_weight: float = 0.05
    reference_temperature_k: float = 293.15
    physics_mode: Literal["coupled_thermoelastic", "simple_heat"] = "coupled_thermoelastic"
    loss_balance_mode: Literal["fixed", "normalize"] = "fixed"
    supervised_loss_scale: float = 1.0
    velocity_loss_scale: float = 1.0
    wave_residual_loss_scale: float = 1.0
    thermal_residual_loss_scale: float = 1.0
    max_grad_norm: float | None = 1.0
    lr_scheduler_patience: int | None = 25
    lr_scheduler_factor: float = 0.5
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    sample_limit: int | None = None
    validation_sample_limit: int | None = None
    seed: int = 42

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["dataset_path"] = str(self.dataset_path)
        payload["output_dir"] = str(self.output_dir)
        payload["val_dataset_path"] = str(self.val_dataset_path) if self.val_dataset_path else None
        return payload
