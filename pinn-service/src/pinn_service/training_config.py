from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


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
    thermal_residual_weight: float = 0.05
    sample_limit: int | None = None
    seed: int = 42

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["dataset_path"] = str(self.dataset_path)
        payload["output_dir"] = str(self.output_dir)
        return payload
