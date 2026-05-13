from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset_registry import dataset_dir, load_scenario, resolve_raw_and_mesh
from src.data.comsol_reader import ALLOWED_RAW_CSV_PREFIXES
from src.training.train import load_config


def parse_args():
    p = argparse.ArgumentParser(description="End-to-end raw COMSOL -> train/fine-tune -> validate -> predict -> visualize")
    p.add_argument("--dataset_id", required=True)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--finetune_config", default="configs/finetune.yaml")
    p.add_argument("--inference_config", default="configs/inference.yaml")
    p.add_argument("--checkpoint", default=None, help="If passed, run fine-tuning from this checkpoint instead of base training")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--clean", action="store_true", help="Clean processed/ and generated outputs. raw/ is never deleted.")
    p.add_argument("--slice_axis", choices=["x", "y", "z"], default="z")
    p.add_argument("--rollout_steps", type=int, default=None)
    p.add_argument("--mode", choices=["full", "decoder_only", "processor_decoder"], default="full")
    return p.parse_args()


def run(cmd: list[str]):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def safe_clean(dataset_id: str, registry_dir: str):
    d = dataset_dir(dataset_id, registry_dir)
    processed = d / "processed"
    if processed.exists():
        shutil.rmtree(processed)
    processed.mkdir(parents=True, exist_ok=True)
    for rel in [
        "outputs/checkpoints", "outputs/checkpoints_finetuned", "outputs/logs", "outputs/logs_finetuned",
        "outputs/metrics", "outputs/predictions", "outputs/validation", "outputs/figures", "outputs/animations",
        "outputs/wave_arrows", "outputs/reports",
    ]:
        p = Path(rel)
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)
        (p / ".gitkeep").touch()


def check_inputs(dataset_id: str, registry_dir: str):
    d = dataset_dir(dataset_id, registry_dir)
    raw_dir, mesh_file = resolve_raw_and_mesh(dataset_id, registry_dir)
    if not d.exists():
        raise FileNotFoundError(f"Dataset directory not found: {d}")
    if not (d / "scenario.yaml").exists():
        raise FileNotFoundError(f"scenario.yaml not found: {d / 'scenario.yaml'}")
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw directory not found: {raw_dir}")
    csvs = [p.name for p in raw_dir.glob("*.csv") if p.stem.lower().startswith(ALLOWED_RAW_CSV_PREFIXES)]
    if not csvs:
        raise FileNotFoundError(f"No supported COMSOL CSV files in {raw_dir}. Expected prefixes: {ALLOWED_RAW_CSV_PREFIXES}")
    scenario = load_scenario(dataset_id, registry_dir)
    print("Input check OK")
    print(f"dataset: {dataset_id}")
    print(f"scenario: {scenario.get('scenario', {}).get('type')} | rock={scenario.get('rock_type')}")
    print(f"raw CSV files: {len(csvs)}")
    if mesh_file:
        print(f"mesh: {mesh_file}")
    else:
        print("mesh: not found -> kNN fallback will be used")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    registry_dir = cfg.get("data", {}).get("registry_dir", "datasets")

    check_inputs(args.dataset_id, registry_dir)
    if args.clean:
        safe_clean(args.dataset_id, registry_dir)

    run([sys.executable, "scripts/prepare_dataset.py", "--config", args.config, "--dataset_id", args.dataset_id])

    if args.checkpoint:
        run([
            sys.executable, "scripts/finetune_model.py", "--config", args.finetune_config,
            "--dataset_id", args.dataset_id, "--checkpoint", args.checkpoint,
            "--epochs", str(args.epochs), "--mode", args.mode,
        ])
        ckpt = "outputs/checkpoints_finetuned/best_model.pt"
    else:
        run([sys.executable, "scripts/train_base_model.py", "--config", args.config, "--dataset_ids", args.dataset_id, "--epochs", str(args.epochs)])
        ckpt = "outputs/checkpoints/best_model.pt"

    run([sys.executable, "scripts/validate_model.py", "--config", args.config, "--dataset_id", args.dataset_id, "--checkpoint", ckpt, "--split", "test", "--slice_axis", args.slice_axis])

    pred_cmd = [sys.executable, "scripts/run_prediction.py", "--config", args.inference_config, "--dataset_id", args.dataset_id, "--checkpoint", ckpt]
    if args.rollout_steps is not None:
        pred_cmd += ["--rollout_steps", str(args.rollout_steps)]
    run(pred_cmd)

    run([sys.executable, "scripts/visualize_results.py", "--dataset_id", args.dataset_id, "--prediction", "outputs/predictions/prediction.pt", "--slice_axis", args.slice_axis])

    important = [
        "outputs/validation/validation_report.html",
        "outputs/wave_arrows/wave_arrows_report.html",
        "outputs/wave_arrows/animations/quiver_temperature_change_velocity.gif",
        "outputs/validation/tables/metrics_per_field.csv",
        "outputs/validation/tables/error_over_time.csv",
    ]
    print("\nOPEN THESE FILES:")
    for p in important:
        print(f"- {p}")


if __name__ == "__main__":
    main()
