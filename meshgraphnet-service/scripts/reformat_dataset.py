from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.universal_formatter import run_universal_format
from src.training.train import load_config


def parse_args():
    p = argparse.ArgumentParser(
        description="Reformat real COMSOL CSV + optional MPHTXT into universal canonical, MeshGraphNet graph, FNO grid, PINN and Transformer adapters."
    )
    p.add_argument("--dataset_id", required=True, help="Dataset folder under datasets/, e.g. sandstone_comsol_real")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--registry_dir", default=None)
    p.add_argument(
        "--formats",
        nargs="+",
        default=["canonical", "graph", "pinn", "transformer"],
        help="Any of: canonical graph fno pinn transformer all. Default skips heavy FNO grid.",
    )
    p.add_argument("--k_nearest", type=int, default=None)
    p.add_argument("--grid_res", nargs=3, type=int, default=[32, 32, 32], metavar=("Z", "Y", "X"))
    p.add_argument(
        "--fno_max_timesteps",
        default="128",
        help="Max timesteps exported to FNO grid. Use 0 or all for full sequence; beware huge files.",
    )
    p.add_argument("--fno_normalization", choices=["raw", "normalized"], default="normalized")
    p.add_argument("--no_legacy_graph", action="store_true", help="Do not copy graph.pt/trajectories.pt to processed root.")
    return p.parse_args()


def _parse_max_timesteps(value: str):
    if str(value).lower() in {"0", "all", "none", "full"}:
        return None
    return int(value)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    if args.registry_dir:
        data_cfg["registry_dir"] = args.registry_dir
    if args.k_nearest is not None:
        data_cfg["k_nearest"] = args.k_nearest

    meta = run_universal_format(
        dataset_id=args.dataset_id,
        config=data_cfg,
        registry_dir=data_cfg.get("registry_dir", "datasets"),
        formats=args.formats,
        grid_resolution=tuple(args.grid_res),
        fno_max_timesteps=_parse_max_timesteps(args.fno_max_timesteps),
        fno_normalization=args.fno_normalization,
        copy_legacy_graph=not args.no_legacy_graph,
    )

    print("✅ Universal dataset prepared")
    for key in [
        "dataset_id", "n_nodes", "n_edges", "n_timesteps", "n_dynamic_fields", "n_static_features",
        "n_mask_features", "node_in_dim", "target_mode", "graph_source",
    ]:
        print(f"{key}: {meta.get(key)}")
    formats = {x.lower() for x in args.formats}
    print("\nCreated/updated under:")
    if "canonical" in formats or "all" in formats:
        print(f"datasets/{args.dataset_id}/processed/canonical")
    if formats & {"graph", "mgn", "meshgraphnet", "all"}:
        print(f"datasets/{args.dataset_id}/processed/graph")
        if not args.no_legacy_graph:
            print(f"datasets/{args.dataset_id}/processed/graph.pt")
            print(f"datasets/{args.dataset_id}/processed/trajectories.pt")
    if "fno" in formats or "all" in formats:
        print(f"datasets/{args.dataset_id}/processed/fno")
    if "pinn" in formats or "all" in formats:
        print(f"datasets/{args.dataset_id}/processed/pinn")
    if formats & {"transformer", "neural_operator_transformer", "all"}:
        print(f"datasets/{args.dataset_id}/processed/transformer")


if __name__ == "__main__":
    main()
