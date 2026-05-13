"""Fine-tuning entry point wrappers."""
from __future__ import annotations

from typing import Dict, List

from .train import train_model


def fine_tune_model(config: Dict, dataset_ids: List[str] | None = None, checkpoint: str | None = None) -> Dict:
    return train_model(config, dataset_ids=dataset_ids, checkpoint_override=checkpoint, fine_tune=True)
