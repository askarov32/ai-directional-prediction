from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import optim

from pinn_service.losses import compute_hybrid_pinn_loss
from pinn_service.model import MLP_PINN
from pinn_service.training_config import TrainingConfig
from pinn_service.training_data import PRIMARY_OUTPUT_NAMES, load_training_data, save_scalers


@dataclass(frozen=True)
class TrainingArtifacts:
    checkpoint_path: Path
    best_checkpoint_path: Path
    metrics_path: Path
    config_path: Path
    scalers_path: Path


def train_pinn(config: TrainingConfig) -> TrainingArtifacts:
    _set_seed(config.seed)

    data = load_training_data(config.dataset_path, sample_limit=config.sample_limit)
    loader = data.make_loader(batch_size=config.batch_size, shuffle=True)

    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device)
    model = MLP_PINN(
        input_dim=len(data.input_feature_names),
        output_dim=len(PRIMARY_OUTPUT_NAMES),
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        activation=config.activation,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    input_mean = torch.tensor(data.input_scaler.mean, dtype=torch.float32, device=device)
    input_std = torch.tensor(data.input_scaler.std, dtype=torch.float32, device=device)
    output_mean = torch.tensor(data.output_scaler.mean, dtype=torch.float32, device=device)
    output_std = torch.tensor(data.output_scaler.std, dtype=torch.float32, device=device)

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict | None = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        aggregates = {
            "supervised_loss": 0.0,
            "velocity_consistency_loss": 0.0,
            "thermal_residual_loss": 0.0,
            "total_loss": 0.0,
        }
        batch_count = 0

        for inputs_scaled, primary_targets_scaled, velocity_targets in loader:
            inputs_scaled = inputs_scaled.to(device)
            primary_targets_scaled = primary_targets_scaled.to(device)
            velocity_targets = velocity_targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss, metrics = compute_hybrid_pinn_loss(
                model=model,
                inputs_scaled=inputs_scaled,
                primary_targets_scaled=primary_targets_scaled,
                velocity_targets=velocity_targets,
                input_scaler_mean=input_mean,
                input_scaler_std=input_std,
                output_scaler_mean=output_mean,
                output_scaler_std=output_std,
                supervised_weight=config.supervised_weight,
                velocity_weight=config.velocity_weight,
                thermal_residual_weight=config.thermal_residual_weight,
            )
            loss.backward()
            optimizer.step()

            for key, value in metrics.items():
                aggregates[key] += value
            batch_count += 1

        epoch_metrics = {key: value / max(batch_count, 1) for key, value in aggregates.items()}
        epoch_metrics["epoch"] = float(epoch)
        history.append(epoch_metrics)

        if epoch_metrics["total_loss"] < best_loss:
            best_loss = epoch_metrics["total_loss"]
            best_state = {
                "model_state_dict": model.state_dict(),
                "config": config.to_dict(),
                "input_feature_names": data.input_feature_names,
                "output_feature_names": PRIMARY_OUTPUT_NAMES,
                "input_scaler": data.input_scaler.to_dict(),
                "output_scaler": data.output_scaler.to_dict(),
                "best_loss": best_loss,
            }

    checkpoint_path = output_dir / "model.pth"
    best_checkpoint_path = output_dir / "best_model.pth"
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint state.")
    torch.save(best_state, checkpoint_path)
    torch.save(best_state, best_checkpoint_path)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps({"history": history, "best_loss": best_loss}, indent=2), encoding="utf-8")

    config_path = output_dir / "training_config.json"
    config_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    scalers_path = save_scalers(output_dir, data)
    return TrainingArtifacts(
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        metrics_path=metrics_path,
        config_path=config_path,
        scalers_path=scalers_path,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
