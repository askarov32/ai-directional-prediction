from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from fno_service.data.dataset import FNOTimeStepDataset, FNOSample, load_fno_grid_tensors
from fno_service.data.preprocessing import FNOChannelConfig
from fno_service.models import FNO2d
from fno_service.training.checkpoints import save_checkpoint, write_json
from fno_service.training.losses import field_mse_loss, relative_l2_loss
from fno_service.training.metrics import finite_metric, mae, relative_l2, rmse


@dataclass(frozen=True)
class FNOTrainingConfig:
    dataset_path: str | Path
    output_dir: str | Path
    device: str = "cpu"
    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    width: int = 32
    modes_x: int = 12
    modes_y: int = 12
    depth: int = 4
    val_fraction: float = 0.2
    sample_limit: int | None = None
    seed: int = 42

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["dataset_path"] = str(Path(self.dataset_path))
        payload["output_dir"] = str(Path(self.output_dir))
        return payload


@dataclass(frozen=True)
class FNOTrainingArtifacts:
    output_dir: Path
    model_path: Path
    best_model_path: Path
    metrics_path: Path
    metrics_csv_path: Path
    training_config_path: Path


def train_fno(config: FNOTrainingConfig) -> FNOTrainingArtifacts:
    _seed_everything(config.seed)
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tensors = load_fno_grid_tensors(config.dataset_path)
    train_indices, val_indices = _split_time_indices(
        total_timesteps=tensors.grid_dynamic.shape[0],
        val_fraction=config.val_fraction,
        sample_limit=config.sample_limit,
        seed=config.seed,
    )
    channel_config = FNOChannelConfig()
    train_dataset = FNOTimeStepDataset(tensors, channel_config=channel_config, time_indices=train_indices)
    val_dataset = FNOTimeStepDataset(tensors, channel_config=channel_config, time_indices=val_indices) if val_indices else None

    first_sample = train_dataset[0]
    _validate_2d_slice(first_sample.inputs)
    in_channels = first_sample.inputs.shape[0]
    out_channels = first_sample.target.shape[0]

    device = torch.device(config.device)
    model = FNO2d(
        in_channels=in_channels,
        out_channels=out_channels,
        width=config.width,
        modes_x=config.modes_x,
        modes_y=config.modes_y,
        depth=config.depth,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=_collate_fno_samples)
    val_loader = (
        DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=_collate_fno_samples)
        if val_dataset is not None
        else None
    )

    model_config = {
        "architecture": "FNO2d",
        "in_channels": in_channels,
        "out_channels": out_channels,
        "width": config.width,
        "modes_x": config.modes_x,
        "modes_y": config.modes_y,
        "depth": config.depth,
    }
    channel_metadata = _build_channel_metadata(tensors, channel_config)
    write_json(output_dir / "training_config.json", config.to_dict())
    write_json(output_dir / "dataset_metadata.json", tensors.metadata)
    write_json(output_dir / "channel_metadata.json", channel_metadata)

    metrics_history: list[dict[str, float | int]] = []
    best_metric = float("inf")
    best_model_path = output_dir / "best_model.pth"
    model_path = output_dir / "model.pth"

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, device=device, optimizer=optimizer)
        val_metrics = _evaluate(model, val_loader, device=device) if val_loader is not None else {}
        selection_metric = float(val_metrics.get("loss", train_metrics["loss"]))

        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": finite_metric(train_metrics["loss"]),
            "train_mae": finite_metric(train_metrics["mae"]),
            "train_rmse": finite_metric(train_metrics["rmse"]),
            "train_relative_l2": finite_metric(train_metrics["relative_l2"]),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if val_metrics:
            row.update(
                {
                    "val_loss": finite_metric(val_metrics["loss"]),
                    "val_mae": finite_metric(val_metrics["mae"]),
                    "val_rmse": finite_metric(val_metrics["rmse"]),
                    "val_relative_l2": finite_metric(val_metrics["relative_l2"]),
                }
            )
        metrics_history.append(row)

        save_checkpoint(
            model_path,
            model=model,
            epoch=epoch,
            metric=selection_metric,
            model_config=model_config,
            training_config=config.to_dict(),
            channel_metadata=channel_metadata,
            dataset_metadata=tensors.metadata,
        )
        if selection_metric < best_metric:
            best_metric = selection_metric
            save_checkpoint(
                best_model_path,
                model=model,
                epoch=epoch,
                metric=best_metric,
                model_config=model_config,
                training_config=config.to_dict(),
                channel_metadata=channel_metadata,
                dataset_metadata=tensors.metadata,
            )

        print(
            f"epoch={epoch}/{config.epochs} "
            f"train_loss={train_metrics['loss']:.6g} "
            f"val_loss={val_metrics.get('loss', float('nan')):.6g} "
            f"best={best_metric:.6g}",
            flush=True,
        )

    metrics_path = write_json(output_dir / "metrics.json", metrics_history)
    metrics_csv_path = _write_metrics_csv(output_dir / "metrics.csv", metrics_history)
    return FNOTrainingArtifacts(
        output_dir=output_dir,
        model_path=model_path,
        best_model_path=best_model_path,
        metrics_path=metrics_path,
        metrics_csv_path=metrics_csv_path,
        training_config_path=output_dir / "training_config.json",
    )


def _run_epoch(model: FNO2d, loader: DataLoader, *, device: torch.device, optimizer: torch.optim.Optimizer) -> dict[str, float]:
    model.train()
    totals = _MetricAccumulator()
    for inputs, target in loader:
        inputs = inputs.to(device)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs)
        loss = field_mse_loss(prediction, target) + 0.01 * relative_l2_loss(prediction, target)
        loss.backward()
        optimizer.step()
        totals.update(prediction.detach(), target.detach(), float(loss.detach().cpu()), batch_size=inputs.shape[0])
    return totals.compute()


@torch.no_grad()
def _evaluate(model: FNO2d, loader: DataLoader | None, *, device: torch.device) -> dict[str, float]:
    if loader is None:
        return {}
    model.eval()
    totals = _MetricAccumulator()
    for inputs, target in loader:
        inputs = inputs.to(device)
        target = target.to(device)
        prediction = model(inputs)
        loss = field_mse_loss(prediction, target) + 0.01 * relative_l2_loss(prediction, target)
        totals.update(prediction, target, float(loss.detach().cpu()), batch_size=inputs.shape[0])
    return totals.compute()


class _MetricAccumulator:
    def __init__(self) -> None:
        self.total_loss = 0.0
        self.total_mae = 0.0
        self.total_rmse = 0.0
        self.total_relative_l2 = 0.0
        self.count = 0

    def update(self, prediction: Tensor, target: Tensor, loss: float, *, batch_size: int) -> None:
        self.total_loss += loss * batch_size
        self.total_mae += mae(prediction, target) * batch_size
        self.total_rmse += rmse(prediction, target) * batch_size
        self.total_relative_l2 += relative_l2(prediction, target) * batch_size
        self.count += batch_size

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            raise ValueError("Cannot compute FNO metrics over an empty loader.")
        return {
            "loss": self.total_loss / self.count,
            "mae": self.total_mae / self.count,
            "rmse": self.total_rmse / self.count,
            "relative_l2": self.total_relative_l2 / self.count,
        }


def _collate_fno_samples(samples: list[FNOSample]) -> tuple[Tensor, Tensor]:
    inputs = np.stack([sample.inputs for sample in samples]).astype(np.float32, copy=False)
    targets = np.stack([sample.target for sample in samples]).astype(np.float32, copy=False)
    _validate_2d_slice(inputs[0])
    return torch.from_numpy(inputs[:, :, 0, :, :]), torch.from_numpy(targets[:, :, 0, :, :])


def _validate_2d_slice(array: np.ndarray) -> None:
    if array.ndim != 4:
        raise ValueError("FNO samples must have shape [channels, Z, Y, X].")
    if array.shape[1] != 1:
        raise ValueError("Current FNO2d training supports only Z=1 grids. Use the converter with --grid-res 1 Y X.")


def _split_time_indices(
    *,
    total_timesteps: int,
    val_fraction: float,
    sample_limit: int | None,
    seed: int,
) -> tuple[list[int], list[int]]:
    max_start = total_timesteps - 1
    if max_start < 1:
        raise ValueError("FNO training requires at least two timesteps.")
    indices = list(range(max_start))
    if sample_limit is not None:
        indices = indices[: max(1, min(sample_limit, len(indices)))]
    random.Random(seed).shuffle(indices)
    if len(indices) == 1:
        return indices, []
    val_count = max(1, int(round(len(indices) * val_fraction))) if val_fraction > 0 else 0
    val_count = min(val_count, len(indices) - 1)
    return indices[val_count:], indices[:val_count]


def _build_channel_metadata(tensors, channel_config: FNOChannelConfig) -> dict:
    input_channels = list(tensors.field_names)
    if channel_config.include_static:
        input_channels.extend(tensors.static_feature_names)
    if channel_config.include_masks:
        input_channels.extend(tensors.mask_names)
    if channel_config.include_coords:
        input_channels.extend(["coord_x", "coord_y", "coord_z"])
    if channel_config.include_time:
        input_channels.append("time_fraction")
    return {
        "input_channels": input_channels,
        "target_channels": list(channel_config.target_channels),
        "field_names": tensors.field_names,
        "static_feature_names": tensors.static_feature_names,
        "mask_names": tensors.mask_names,
    }


def _write_metrics_csv(path: Path, rows: list[dict[str, float | int]]) -> Path:
    if not rows:
        raise ValueError("No FNO metrics rows were produced.")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
