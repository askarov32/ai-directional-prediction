from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset_registry import register_dataset
from src.data.scenario import default_scenario


def parse_args():
    p = argparse.ArgumentParser(description="Register real COMSOL dataset with scenario.yaml")
    p.add_argument("--dataset_id", required=True)
    p.add_argument("--raw_dir", required=True)
    p.add_argument("--mesh_file", required=True)
    p.add_argument("--registry_dir", default="datasets")
    p.add_argument("--rock_type", default="unknown")
    p.add_argument("--physics_type", default="thermoelastic_wave")
    p.add_argument("--source_type", default="heated_rod")
    p.add_argument("--initial_temperature", type=float, default=773.15)
    p.add_argument("--background_temperature", type=float, default=293.15)
    p.add_argument("--source_radius", type=float, default=0.01)
    p.add_argument("--source_center", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    p.add_argument("--dt", type=float, default=1e-4)
    p.add_argument("--young_modulus", type=float, default=0.0)
    p.add_argument("--poisson_ratio", type=float, default=0.0)
    p.add_argument("--density", type=float, default=0.0)
    p.add_argument("--thermal_expansion", type=float, default=0.0)
    p.add_argument("--thermal_conductivity", type=float, default=0.0)
    p.add_argument("--heat_capacity", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()
    sc = default_scenario(args.dataset_id)
    sc["rock_type"] = args.rock_type
    sc["physics"]["type"] = args.physics_type
    sc["source"].update({
        "type": args.source_type,
        "initial_temperature": args.initial_temperature,
        "background_temperature": args.background_temperature,
        "center": args.source_center,
        "radius": args.source_radius,
    })
    sc["time"]["step"] = args.dt
    sc["material"].update({
        "young_modulus": args.young_modulus,
        "poisson_ratio": args.poisson_ratio,
        "density": args.density,
        "thermal_expansion": args.thermal_expansion,
        "thermal_conductivity": args.thermal_conductivity,
        "heat_capacity": args.heat_capacity,
    })
    d = register_dataset(args.dataset_id, args.raw_dir, args.mesh_file, args.registry_dir, sc)
    print(f"✅ Registered dataset: {d}")
    print(f"   Scenario: {d / 'scenario.yaml'}")


if __name__ == "__main__":
    main()
