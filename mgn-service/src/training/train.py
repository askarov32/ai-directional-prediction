"""Training utilities for variable-mesh datasets."""
from __future__ import annotations

import json
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.data.dataset_registry import list_dataset_ids
from src.data.pipeline import load_processed_dataset
from src.models.losses import WeightedFieldMSE, compute_metrics
from src.models.meshgraphnet import ConditionalMeshGraphNet
from .checkpoint_manager import load_checkpoint, save_checkpoint


def load_config(path: str | Path) -> Dict:
    import yaml
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "include" in cfg:
        base_path = Path(cfg["include"])
        if not base_path.is_absolute():
            base_path = Path.cwd() / base_path
        if base_path.exists():
            base = load_config(base_path)
            cfg = deep_update(base, {k: v for k, v in cfg.items() if k != "include"})
    return cfg


def deep_update(a: Dict, b: Dict) -> Dict:
    out = dict(a or {})
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def setup_device(device_cfg: str = "auto") -> torch.device:
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)
    if device.type == "cpu":
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
    print("=" * 60)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        print(f"CUDA:   {torch.version.cuda}")
    print("=" * 60)
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Ограничиваем CPU threads: на маленьких/средних графах это часто стабильнее
    # и не даёт OpenMP/MKL подвисать на некоторых Windows/CPU окружениях.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(use_amp: bool):
    if use_amp:
        try:
            return torch.amp.autocast(device_type="cuda", enabled=True)
        except Exception:
            return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


def load_training_datasets(dataset_ids: List[str] | None, registry_dir: str | Path = "datasets") -> Dict:
    if not dataset_ids:
        dataset_ids = list_dataset_ids(registry_dir, require_processed=True)
    if not dataset_ids:
        raise ValueError("No processed datasets found. Run prepare_dataset.py first.")

    all_splits = {"train": [], "val": [], "test": []}
    metadatas = []
    for ds_id in dataset_ids:
        ds = load_processed_dataset(ds_id, registry_dir)
        graph = ds["graph"]
        md = ds["metadata"]
        metadatas.append(md)
        for split in ["train", "val", "test"]:
            for x, y in ds["data"].get(split, []):
                all_splits[split].append(
                    {
                        "x": x,
                        "y": y,
                        "edge_index": graph["edge_index"],
                        "edge_attr": graph["edge_attr"],
                        "dataset_id": ds_id,
                    }
                )
    field_names = metadatas[0]["field_names"]
    node_in_dim = metadatas[0]["node_in_dim"]
    edge_in_dim = metadatas[0]["edge_in_dim"]
    out_dim = len(field_names)
    for md in metadatas[1:]:
        if md["field_names"] != field_names:
            raise ValueError(
                f"Dataset {md['dataset_id']} fields differ from first dataset. "
                "For multi-dataset training, export the same field set from COMSOL."
            )
        if md["node_in_dim"] != node_in_dim:
            raise ValueError("node_in_dim differs. Ensure scenario feature schema is consistent.")
    return {
        "splits": all_splits,
        "field_names": field_names,
        "node_in_dim": node_in_dim,
        "edge_in_dim": edge_in_dim,
        "out_dim": out_dim,
        "metadatas": metadatas,
        "dataset_ids": dataset_ids,
    }


def build_model(config: Dict, node_in_dim: int, edge_in_dim: int, out_dim: int) -> ConditionalMeshGraphNet:
    mcfg = config.get("model", {})
    return ConditionalMeshGraphNet(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        out_dim=out_dim,
        latent_dim=int(mcfg.get("latent_dim", 128)),
        message_passing_steps=int(mcfg.get("message_passing_steps", 10)),
        mlp_layers=int(mcfg.get("mlp_layers", 3)),
        dropout=float(mcfg.get("dropout", 0.05)),
        layer_norm=bool(mcfg.get("layer_norm", True)),
    )


def move_sample(sample: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sample.items()}


def evaluate_loss(model, loss_fn, samples: List[Dict], device: torch.device, use_amp: bool = False) -> float:
    if not samples:
        return float("inf")
    model.eval()
    total = 0.0
    with torch.no_grad():
        for s in samples:
            s = move_sample(s, device)
            with autocast_context(use_amp):
                pred = model(s["x"], s["edge_index"], s["edge_attr"])
                loss = loss_fn(pred, s["y"])
            total += float(loss.detach().cpu())
    return total / len(samples)


def train_model(config: Dict, dataset_ids: List[str] | None = None, checkpoint_override: str | None = None, fine_tune: bool = False) -> Dict:
    training = config.get("training", {})
    data_cfg = config.get("data", {})
    registry_dir = data_cfg.get("registry_dir", "datasets")
    set_seed(int(training.get("seed", 42)))
    device = setup_device(training.get("device", "auto"))

    bundle = load_training_datasets(dataset_ids, registry_dir)
    samples = bundle["splits"]
    if not samples["train"]:
        raise ValueError("Train split is empty.")

    model = build_model(config, bundle["node_in_dim"], bundle["edge_in_dim"], bundle["out_dim"]).to(device)
    if fine_tune:
        ft_cfg = config.get("fine_tune", {}) or {}
        mode = ft_cfg.get("mode", training.get("fine_tune_mode", "full"))
        model.freeze_for_finetune(mode)
        print(f"Fine-tune mode: {mode}")
    elif training.get("freeze_encoder_processor", False):
        model.freeze_encoder_processor()

    loss_fn = WeightedFieldMSE(bundle["field_names"], config.get("loss_weights", {}))
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(training.get("lr", 3e-4)),
        weight_decay=float(training.get("weight_decay", 1e-6)),
    )

    resume = checkpoint_override or training.get("resume_checkpoint", "")
    start_epoch = 1
    best_val = float("inf")
    if resume:
        resume_path = Path(resume)
        if resume_path.exists():
            print(f"Loading checkpoint: {resume_path}")
            ckpt = load_checkpoint(resume_path, model, optimizer=None, map_location=device, strict=True)
            start_epoch = int(ckpt.get("epoch", 0)) + 1 if not fine_tune else 1
            best_val = float(ckpt.get("val_loss") or best_val)
        else:
            print(f"WARNING: checkpoint not found: {resume_path}")

    use_amp = bool(training.get("use_amp", True)) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    epochs = int(training.get("epochs", 200))
    grad_clip = float(training.get("grad_clip", 1.0))
    patience = int(training.get("early_stopping_patience", 30))
    ckpt_dir = Path(training.get("checkpoint_dir", "outputs/checkpoints"))
    log_dir = Path(training.get("log_dir", "outputs/logs"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Datasets: {bundle['dataset_ids']}")
    print(f"Samples: train={len(samples['train'])} val={len(samples['val'])} test={len(samples['test'])}")
    print(f"Dims: node_in={bundle['node_in_dim']} edge_in={bundle['edge_in_dim']} out={bundle['out_dim']}")
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable")

    history = []
    no_improve = 0
    tag = "best"
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        random.shuffle(samples["train"])
        train_loss = 0.0
        for s in tqdm(samples["train"], desc=f"epoch {epoch}/{epochs}", leave=False):
            s = move_sample(s, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(use_amp):
                pred = model(s["x"], s["edge_index"], s["edge_attr"])
                loss = loss_fn(pred, s["y"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach().cpu())
        train_loss /= max(len(samples["train"]), 1)
        val_loss = evaluate_loss(model, loss_fn, samples["val"], device, use_amp)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch:04d} train={train_loss:.6g} val={val_loss:.6g}")

        metadata = {
            "field_names": bundle["field_names"],
            "node_in_dim": bundle["node_in_dim"],
            "edge_in_dim": bundle["edge_in_dim"],
            "out_dim": bundle["out_dim"],
            "dataset_ids": bundle["dataset_ids"],
            "config_model": config.get("model", {}),
            "target_mode": bundle["metadatas"][0].get("target_mode", "delta"),
        }
        save_checkpoint(ckpt_dir / "last_model.pt", model, optimizer, epoch, val_loss, metadata)
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            save_checkpoint(ckpt_dir / f"{tag}_model.pt", model, optimizer, epoch, val_loss, metadata)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    history_file = log_dir / "train_history.json"
    with history_file.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    if fine_tune:
        with (log_dir / "finetune_history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    test_loss = evaluate_loss(model, loss_fn, samples["test"], device, use_amp)
    metrics_file = log_dir / "test_metrics.json"
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump({"test_loss": test_loss, "best_val_loss": best_val}, f, indent=2, ensure_ascii=False)
    if fine_tune:
        with (log_dir / "finetune_test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump({"test_loss": test_loss, "best_val_loss": best_val}, f, indent=2, ensure_ascii=False)
    return {"best_val_loss": best_val, "test_loss": test_loss, "checkpoint": str(ckpt_dir / "best_model.pt")}
