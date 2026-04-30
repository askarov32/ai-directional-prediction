from __future__ import annotations

import argparse
from pathlib import Path

from pinn_service.dataset_builder import build_dataset_from_exports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare unified PINN-ready datasets from COMSOL thermoelastic CSV exports.",
    )
    parser.add_argument("--materials", required=True, help="Path to data_materials.csv")
    parser.add_argument("--temperature", required=True, help="Path to data_temperature.csv")
    parser.add_argument("--displacement", required=True, help="Path to data_displacement.csv")
    parser.add_argument("--stress1", required=True, help="Path to data_stress_1.csv")
    parser.add_argument("--stress2", required=True, help="Path to data_stress_2.csv")
    parser.add_argument("--stress3", required=True, help="Path to data_stress_3.csv")
    parser.add_argument("--output-dir", required=True, help="Directory where processed artifacts will be written")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "float64"),
        help="Output dtype for the generated arrays.",
    )
    parser.add_argument(
        "--build-training-matrix",
        action="store_true",
        help="Also create flattened inputs/targets for supervised or hybrid PINN experiments.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = build_dataset_from_exports(
        materials_path=args.materials,
        temperature_path=args.temperature,
        displacement_path=args.displacement,
        stress1_path=args.stress1,
        stress2_path=args.stress2,
        stress3_path=args.stress3,
        output_dir=args.output_dir,
        dtype=args.dtype,
        build_training_matrix=args.build_training_matrix,
    )

    print("Structured dataset:", artifacts.structured_path)
    print("Metadata:", artifacts.metadata_path)
    if artifacts.training_matrix_path is not None:
        print("Training matrix:", artifacts.training_matrix_path)


if __name__ == "__main__":
    main()
