from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from transformer_service.dataset import (
    INPUT_CHANNEL_NAMES,
    TARGET_CHANNEL_NAMES,
    AutoregressivePairsDataset,
    build_train_val_split,
    load_pairs_bundle,
)
from transformer_service.losses import supervised_mse
from transformer_service.model import OFormer
from transformer_service.training_config import TrainingConfig


@dataclass(frozen=True)
class TrainingArtifacts:
    checkpoint_path: Path
    best_checkpoint_path: Path
    metrics_path: Path
    config_path: Path
    scalers_path: Path


def train_transformer(config: TrainingConfig) -> TrainingArtifacts:
    _set_seed(config.seed)

    bundle = load_pairs_bundle(config.dataset_path)
    train_indices, val_indices = build_train_val_split(bundle, train_fraction=0.8)

    train_dataset = AutoregressivePairsDataset(
        bundle, train_indices, n_tokens=config.n_tokens, seed=config.seed
    )
    val_dataset = AutoregressivePairsDataset(
        bundle, val_indices, n_tokens=None, seed=config.seed
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device)
    model = OFormer(
        input_dim=len(bundle.input_channel_names),
        query_dim=3,
        output_dim=len(bundle.target_channel_names),
        d_model=config.d_model,
        n_heads=config.n_heads,
        enc_depth=config.enc_depth,
        dec_depth=config.dec_depth,
        ffn_expansion=config.ffn_expansion,
        dropout=config.dropout,
        activation=config.activation,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict | None = None
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_aggregate = _zero_aggregate()
        train_batches = 0
        for batch in train_loader:
            input_tokens = batch["input_tokens"].to(device)
            query_coords = batch["query_coords"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(input_tokens, query_coords)
            loss, metrics = supervised_mse(pred, target)
            loss.backward()
            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()

            for key, value in metrics.items():
                train_aggregate[key] = train_aggregate.get(key, 0.0) + value
            train_batches += 1

        model.eval()
        val_aggregate = _zero_aggregate()
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                input_tokens = batch["input_tokens"].to(device)
                query_coords = batch["query_coords"].to(device)
                target = batch["target"].to(device)
                pred = model(input_tokens, query_coords)
                _, metrics = supervised_mse(pred, target)
                for key, value in metrics.items():
                    val_aggregate[key] = val_aggregate.get(key, 0.0) + value
                val_batches += 1

        train_metrics = {f"train_{k}": v / max(train_batches, 1) for k, v in train_aggregate.items()}
        val_metrics = {f"val_{k}": v / max(val_batches, 1) for k, v in val_aggregate.items()}
        epoch_record = {"epoch": float(epoch), "lr": optimizer.param_groups[0]["lr"]}
        epoch_record.update(train_metrics)
        epoch_record.update(val_metrics)
        history.append(epoch_record)

        scheduler.step()

        val_total = epoch_record.get("val_total_loss", float("inf"))
        if val_total < best_loss:
            best_loss = val_total
            best_state = {
                "model_state_dict": model.state_dict(),
                "config": config.to_dict(),
                "input_channel_names": list(bundle.input_channel_names),
                "target_channel_names": list(bundle.target_channel_names),
                "input_mean": bundle.input_mean.tolist(),
                "input_std": bundle.input_std.tolist(),
                "target_mean": bundle.target_mean.tolist(),
                "target_std": bundle.target_std.tolist(),
                "best_loss": float(best_loss),
                "best_epoch": int(epoch),
                "coords": bundle.coords.tolist(),
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if config.early_stop_patience > 0 and patience_counter >= config.early_stop_patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint state.")

    checkpoint_path = output_dir / "model.pth"
    best_checkpoint_path = output_dir / "best_model.pth"
    torch.save(best_state, checkpoint_path)
    torch.save(best_state, best_checkpoint_path)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps({"history": history, "best_loss": float(best_loss)}, indent=2),
        encoding="utf-8",
    )

    config_path = output_dir / "training_config.json"
    config_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    scalers_path = output_dir / "scalers.json"
    scalers_path.write_text(
        json.dumps(
            {
                "input_channel_names": list(bundle.input_channel_names),
                "input_scaler": {
                    "mean": bundle.input_mean.tolist(),
                    "std": bundle.input_std.tolist(),
                },
                "target_channel_names": list(bundle.target_channel_names),
                "target_scaler": {
                    "mean": bundle.target_mean.tolist(),
                    "std": bundle.target_std.tolist(),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return TrainingArtifacts(
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        metrics_path=metrics_path,
        config_path=config_path,
        scalers_path=scalers_path,
    )


def _zero_aggregate() -> dict[str, float]:
    base = {"total_loss": 0.0}
    for name in TARGET_CHANNEL_NAMES:
        base[f"mse_{name}"] = 0.0
    return base


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Reference to avoid unused-import warnings; INPUT_CHANNEL_NAMES is documented public surface.
_ = INPUT_CHANNEL_NAMES
