from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    dataset_path: Path
    output_dir: Path
    device: str = "cpu"
    epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    d_model: int = 128
    n_heads: int = 4
    enc_depth: int = 4
    dec_depth: int = 4
    ffn_expansion: int = 4
    dropout: float = 0.1
    activation: str = "gelu"
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 30
    seed: int = 42
    n_tokens: int | None = 1024

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["dataset_path"] = str(self.dataset_path)
        payload["output_dir"] = str(self.output_dir)
        return payload
