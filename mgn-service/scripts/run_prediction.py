from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Стабильнее на CPU/Windows/малых графах.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    p = argparse.ArgumentParser(description="Run autoregressive prediction and visualization")
    p.add_argument("--config", default="configs/inference.yaml")
    p.add_argument("--dataset_id", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--rollout_steps", type=int, default=None)
    p.add_argument("--initial_state_source", choices=["from_dataset", "nearest_scenario", "user_defined"], default=None)
    p.add_argument("--no_animate", action="store_true")
    p.add_argument("--no_vtk", action="store_true")
    p.add_argument("--no_plots", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    import torch
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    from src.training.train import load_config
    from src.inference.rollout import run_rollout
    from src.inference.export import compute_derived_fields, export_prediction

    cfg = load_config(args.config)
    if args.initial_state_source:
        cfg.setdefault("inference", {})["initial_state_source"] = args.initial_state_source

    traj, coords, field_names, times, md = run_rollout(cfg, args.dataset_id, args.checkpoint, args.rollout_steps)
    inf = cfg.get("inference", {})
    derived = compute_derived_fields(traj, field_names, float(inf.get("risk_threshold", 0.8)))
    paths = export_prediction(traj, coords, field_names, times, derived, inf.get("output_dir", "outputs/predictions"))

    if not args.no_plots:
        from src.visualization.plot_3d import plot_snapshots, plot_time_series
        plot_snapshots(traj, coords, field_names, derived, inf.get("figures_dir", "outputs/figures"))
        plot_time_series(traj, field_names, times, inf.get("figures_dir", "outputs/figures"))

    if inf.get("animate", True) and not args.no_animate:
        from src.visualization.animation import animate_main_fields
        animate_main_fields(traj, coords, field_names, derived, inf.get("animation_dir", "outputs/animations"))

    if inf.get("export_vtk", True) and not args.no_vtk:
        from src.visualization.vtk_export import export_vtk_sequence
        export_vtk_sequence(coords, traj, field_names, derived, Path(inf.get("output_dir", "outputs/predictions")) / "vtk")

    print("✅ Prediction complete")
    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
