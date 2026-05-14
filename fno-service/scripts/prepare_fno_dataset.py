from __future__ import annotations

import argparse
from pathlib import Path

from fno_service.data.dataset import load_fno_grid_tensors
from fno_service.data.pinn_to_grid import convert_pinn_structured_to_fno_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare or validate FNO grid datasets.")
    parser.add_argument("--source", default=None, help="Existing universal formatter FNO dataset directory.")
    parser.add_argument("--pinn-structured", default=None, help="PINN structured_dataset.npz fallback source.")
    parser.add_argument("--pinn-metadata", default=None, help="PINN dataset_metadata.json fallback metadata.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--grid-res", nargs=3, type=int, default=[8, 16, 16], metavar=("Z", "Y", "X"))
    parser.add_argument("--max-timesteps", type=int, default=64)
    parser.add_argument("--validate", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.source:
        dataset_dir = Path(args.source).expanduser().resolve()
    elif args.pinn_structured:
        convert_pinn_structured_to_fno_grid(
            structured_path=args.pinn_structured,
            metadata_path=args.pinn_metadata,
            output_dir=output_dir,
            grid_resolution=tuple(args.grid_res),
            max_timesteps=args.max_timesteps,
        )
        dataset_dir = output_dir
    else:
        raise ValueError("Provide either --source or --pinn-structured.")

    if args.validate:
        tensors = load_fno_grid_tensors(dataset_dir)
        print("FNO dataset:", dataset_dir)
        print("grid_dynamic:", tensors.grid_dynamic.shape)
        print("grid_static:", tensors.grid_static.shape)
        print("grid_masks:", tensors.grid_masks.shape)
        print("grid_coords:", tensors.grid_coords.shape)


if __name__ == "__main__":
    main()
