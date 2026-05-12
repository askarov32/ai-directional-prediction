from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class TrainingConfig:
    dataset_path: Path
    output_dir: Path
    device: str = "cpu"
    epochs: int = 25
    batch_size: int = 4096
    learning_rate: float = 1e-3
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
    max_grad_norm: float | None = 1.0
    sample_limit: int | None = None
    seed: int = 42

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["dataset_path"] = str(self.dataset_path)
        payload["output_dir"] = str(self.output_dir)
        return payload
