from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from pinn_service.losses import compute_hybrid_pinn_loss
from pinn_service.model import MLP_PINN
from pinn_service.training_config import TrainingConfig
from pinn_service.training_data import LoadedTrainingData, PRIMARY_OUTPUT_NAMES, load_training_data, save_scalers


@dataclass(frozen=True)
class TrainingArtifacts:
    checkpoint_path: Path
    best_checkpoint_path: Path
    metrics_path: Path
    metrics_csv_path: Path
    config_path: Path
    scalers_path: Path


METRIC_FIELDS = [
    "epoch",
    "learning_rate",
    "grad_norm",
    "epochs_without_improvement",
    "supervised_loss",
    "velocity_consistency_loss",
    "wave_residual_loss",
    "thermal_residual_loss",
    "normalized_supervised_loss",
    "normalized_velocity_consistency_loss",
    "normalized_wave_residual_loss",
    "normalized_thermal_residual_loss",
    "total_loss",
    "val_supervised_loss",
    "val_velocity_consistency_loss",
    "val_wave_residual_loss",
    "val_thermal_residual_loss",
    "val_normalized_supervised_loss",
    "val_normalized_velocity_consistency_loss",
    "val_normalized_wave_residual_loss",
    "val_normalized_thermal_residual_loss",
    "val_total_loss",
    "best_metric",
    "best_so_far",
]


def train_pinn(config: TrainingConfig) -> TrainingArtifacts:
    _set_seed(config.seed)

    data = load_training_data(config.dataset_path, sample_limit=config.sample_limit, seed=config.seed)
    loader = data.make_loader(batch_size=config.batch_size, shuffle=True)
    validation_data = _load_validation_data(config, data)
    validation_loader = (
        validation_data.make_loader(
            batch_size=config.validation_batch_size or config.batch_size,
            shuffle=False,
        )
        if validation_data
        else None
    )

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
    scheduler = _build_scheduler(optimizer, config)

    input_mean = torch.tensor(data.input_scaler.mean, dtype=torch.float32, device=device)
    input_std = torch.tensor(data.input_scaler.std, dtype=torch.float32, device=device)
    output_mean = torch.tensor(data.output_scaler.mean, dtype=torch.float32, device=device)
    output_std = torch.tensor(data.output_scaler.std, dtype=torch.float32, device=device)
    loss_context = LossContext(
        input_mean=input_mean,
        input_std=input_std,
        output_mean=output_mean,
        output_std=output_std,
    )

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict | None = None
    best_metric_name = "val_total_loss" if validation_loader else "total_loss"
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        aggregates = _empty_loss_aggregates()
        aggregates["grad_norm"] = 0.0
        batch_count = 0

        for inputs_scaled, primary_targets_scaled, velocity_targets in loader:
            inputs_scaled = inputs_scaled.to(device)
            primary_targets_scaled = primary_targets_scaled.to(device)
            velocity_targets = velocity_targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss, metrics = compute_loss_for_batch(
                model=model,
                inputs_scaled=inputs_scaled,
                primary_targets_scaled=primary_targets_scaled,
                velocity_targets=velocity_targets,
                context=loss_context,
                config=config,
            )
            loss.backward()
            grad_norm = _clip_gradients(model, config.max_grad_norm)
            optimizer.step()

            for key, value in metrics.items():
                aggregates[key] += value
            aggregates["grad_norm"] += grad_norm
            batch_count += 1

        epoch_metrics = {key: value / max(batch_count, 1) for key, value in aggregates.items()}
        epoch_metrics["epoch"] = float(epoch)
        epoch_metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        if validation_loader:
            validation_metrics = evaluate_loss(
                model=model,
                loader=validation_loader,
                device=device,
                context=loss_context,
                config=config,
                prefix="val_",
            )
            epoch_metrics.update(validation_metrics)
        epoch_metrics["best_metric"] = epoch_metrics[best_metric_name]
        improved = epoch_metrics[best_metric_name] < (best_loss - config.early_stopping_min_delta)
        if improved:
            best_loss = epoch_metrics[best_metric_name]
            epochs_without_improvement = 0
            best_state = _snapshot_checkpoint_state(
                model=model,
                config=config,
                data=data,
                best_loss=best_loss,
                best_metric_name=best_metric_name,
                best_epoch=epoch,
            )
        else:
            epochs_without_improvement += 1
        epoch_metrics["epochs_without_improvement"] = float(epochs_without_improvement)
        epoch_metrics["best_so_far"] = float(best_loss)
        history.append(epoch_metrics)

        if scheduler is not None:
            scheduler.step(epoch_metrics[best_metric_name])
        if _should_stop_early(config, epochs_without_improvement):
            break

    checkpoint_path = output_dir / "model.pth"
    best_checkpoint_path = output_dir / "best_model.pth"
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint state.")
    latest_state = _snapshot_checkpoint_state(
        model=model,
        config=config,
        data=data,
        best_loss=history[-1]["total_loss"] if history else float("inf"),
        best_metric_name="total_loss",
        best_epoch=int(history[-1]["epoch"]) if history else config.epochs,
    )
    torch.save(latest_state, checkpoint_path)
    torch.save(best_state, best_checkpoint_path)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "history": history,
                "best_loss": best_loss,
                "best_metric": best_metric_name,
                "validation_enabled": validation_loader is not None,
                "stopped_early": _did_stop_early(config, history),
                "completed_epochs": len(history),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics_csv_path = output_dir / "metrics.csv"
    _write_metrics_csv(metrics_csv_path, history)

    config_path = output_dir / "training_config.json"
    config_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    scalers_path = save_scalers(output_dir, data)
    return TrainingArtifacts(
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        metrics_path=metrics_path,
        metrics_csv_path=metrics_csv_path,
        config_path=config_path,
        scalers_path=scalers_path,
    )


@dataclass(frozen=True)
class LossContext:
    input_mean: torch.Tensor
    input_std: torch.Tensor
    output_mean: torch.Tensor
    output_std: torch.Tensor


def _load_validation_data(config: TrainingConfig, data: LoadedTrainingData) -> LoadedTrainingData | None:
    if config.val_dataset_path is None:
        return None
    validation_data = load_training_data(
        config.val_dataset_path,
        sample_limit=config.validation_sample_limit,
        seed=config.seed,
        input_scaler=data.input_scaler,
        output_scaler=data.output_scaler,
    )
    if validation_data.input_feature_names != data.input_feature_names:
        raise ValueError("Validation dataset input features must match the training dataset input features.")
    if validation_data.target_feature_names != data.target_feature_names:
        raise ValueError("Validation dataset target features must match the training dataset target features.")
    return validation_data


def _empty_loss_aggregates() -> dict[str, float]:
    return {
        "supervised_loss": 0.0,
        "velocity_consistency_loss": 0.0,
        "wave_residual_loss": 0.0,
        "thermal_residual_loss": 0.0,
        "normalized_supervised_loss": 0.0,
        "normalized_velocity_consistency_loss": 0.0,
        "normalized_wave_residual_loss": 0.0,
        "normalized_thermal_residual_loss": 0.0,
        "total_loss": 0.0,
    }


def compute_loss_for_batch(
    *,
    model: torch.nn.Module,
    inputs_scaled: torch.Tensor,
    primary_targets_scaled: torch.Tensor,
    velocity_targets: torch.Tensor,
    context: LossContext,
    config: TrainingConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    return compute_hybrid_pinn_loss(
        model=model,
        inputs_scaled=inputs_scaled,
        primary_targets_scaled=primary_targets_scaled,
        velocity_targets=velocity_targets,
        input_scaler_mean=context.input_mean,
        input_scaler_std=context.input_std,
        output_scaler_mean=context.output_mean,
        output_scaler_std=context.output_std,
        supervised_weight=config.supervised_weight,
        velocity_weight=config.velocity_weight,
        wave_residual_weight=config.wave_residual_weight,
        thermal_residual_weight=config.thermal_residual_weight,
        reference_temperature_k=config.reference_temperature_k,
        physics_mode=config.physics_mode,
        loss_balance_mode=config.loss_balance_mode,
        supervised_loss_scale=config.supervised_loss_scale,
        velocity_loss_scale=config.velocity_loss_scale,
        wave_residual_loss_scale=config.wave_residual_loss_scale,
        thermal_residual_loss_scale=config.thermal_residual_loss_scale,
    )


def evaluate_loss(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    context: LossContext,
    config: TrainingConfig,
    prefix: str = "",
) -> dict[str, float]:
    model.eval()
    aggregates = _empty_loss_aggregates()
    batch_count = 0

    for inputs_scaled, primary_targets_scaled, velocity_targets in loader:
        loss, metrics = compute_loss_for_batch(
            model=model,
            inputs_scaled=inputs_scaled.to(device),
            primary_targets_scaled=primary_targets_scaled.to(device),
            velocity_targets=velocity_targets.to(device),
            context=context,
            config=config,
        )
        del loss
        for key, value in metrics.items():
            aggregates[key] += value
        batch_count += 1

    return {f"{prefix}{key}": value / max(batch_count, 1) for key, value in aggregates.items()}


def _snapshot_checkpoint_state(
    *,
    model: torch.nn.Module,
    config: TrainingConfig,
    data: LoadedTrainingData,
    best_loss: float,
    best_metric_name: str,
    best_epoch: int,
) -> dict:
    return {
        "model_state_dict": _clone_state_dict(model.state_dict()),
        "config": config.to_dict(),
        "input_feature_names": data.input_feature_names,
        "output_feature_names": PRIMARY_OUTPUT_NAMES,
        "input_scaler": data.input_scaler.to_dict(),
        "output_scaler": data.output_scaler.to_dict(),
        "best_loss": best_loss,
        "best_metric": best_metric_name,
        "best_epoch": best_epoch,
    }


def _clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def _build_scheduler(optimizer: optim.Optimizer, config: TrainingConfig) -> optim.lr_scheduler.ReduceLROnPlateau | None:
    if config.lr_scheduler_patience is None or config.lr_scheduler_patience <= 0:
        return None
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
        min_lr=config.min_learning_rate,
    )


def _should_stop_early(config: TrainingConfig, epochs_without_improvement: int) -> bool:
    return bool(
        config.early_stopping_patience is not None
        and config.early_stopping_patience > 0
        and epochs_without_improvement >= config.early_stopping_patience
    )


def _did_stop_early(config: TrainingConfig, history: list[dict[str, float]]) -> bool:
    if not history:
        return False
    return _should_stop_early(config, int(history[-1].get("epochs_without_improvement", 0.0)))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clip_gradients(model: torch.nn.Module, max_grad_norm: float | None) -> float:
    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
    if not parameters:
        return 0.0
    if max_grad_norm is not None and max_grad_norm > 0:
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
        return float(norm.detach().cpu())

    total = torch.zeros((), device=parameters[0].grad.device)
    for parameter in parameters:
        total = total + parameter.grad.detach().pow(2).sum()
    return float(torch.sqrt(total).cpu())


def _write_metrics_csv(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in history:
            writer.writerow({field: row.get(field, "") for field in METRIC_FIELDS})
