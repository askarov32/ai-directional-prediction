#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_CATALOG = Path("backend/data/media/catalog.json")
DEFAULT_OUTPUT = Path("artifacts/data_experiments/inputs/model_comparison_inputs.jsonl")
DEFAULT_METADATA_OUTPUT = Path(
    "artifacts/data_experiments/inputs/model_comparison_inputs.metadata.json"
)

DEFAULT_BOUNDARY_CONDITIONS = {
    "left": "fixed",
    "right": "free",
    "top": "insulated",
    "bottom": "insulated",
}

DEFAULT_SOURCE = {
    "type": "thermal_pulse",
    "x": 0.15,
    "y": 0.40,
    "z": 0.0,
    "amplitude": 1.0,
    "frequency_hz": 50.0,
    "direction": [1.0, 0.0, 0.0],
}

DEFAULT_PROBE = {
    "x": 0.70,
    "y": 0.55,
    "z": 0.0,
}

DEFAULT_DOMAIN = {
    "type": "rect_2d",
    "size": {
        "lx": 1.0,
        "ly": 1.0,
        "lz": 0.0,
    },
    "resolution": {
        "nx": 128,
        "ny": 128,
        "nz": 1,
    },
    "boundary_conditions": DEFAULT_BOUNDARY_CONDITIONS,
}


@dataclass(frozen=True)
class ExperimentCase:
    case_id: str
    material: str
    medium_id: str
    temperature_c: float
    pressure_mpa: float
    time_ms: float
    frequency_hz: float
    boundary_conditions: dict[str, Any]
    input: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reproducible canonical inputs for direct model-service comparison."
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
        help="Path to the backend media catalog JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the JSONL output file.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=DEFAULT_METADATA_OUTPUT,
        help="Path to the JSON metadata sidecar file.",
    )
    parser.add_argument(
        "--materials",
        nargs="+",
        default=["sandstone_medium", "basalt"],
        help="Medium ids to include in the experiment grid.",
    )
    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=[20.0, 80.0, 140.0, 220.0, 300.0],
        help="Scenario temperatures in Celsius.",
    )
    parser.add_argument(
        "--pressures",
        nargs="+",
        type=float,
        default=[5.0, 35.0],
        help="Scenario pressures in MPa.",
    )
    parser.add_argument(
        "--time-ms",
        nargs="+",
        type=float,
        default=[6.0, 12.0],
        help="Scenario time points in milliseconds.",
    )
    parser.add_argument(
        "--frequency-hz",
        type=float,
        default=50.0,
        help="Shared source frequency for all generated cases.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for deterministic ordering.",
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=None,
        help="Optional cap on the number of generated cases after deterministic shuffling.",
    )
    return parser.parse_args()


def load_catalog(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Catalog at {path} must be a JSON array.")
    return payload


def resolve_media(catalog: list[dict[str, Any]], medium_ids: list[str]) -> list[dict[str, Any]]:
    by_id = {entry["id"]: entry for entry in catalog if isinstance(entry, dict) and "id" in entry}
    missing = [medium_id for medium_id in medium_ids if medium_id not in by_id]
    if missing:
        raise KeyError(f"Unknown medium ids in catalog: {missing}")
    return [by_id[medium_id] for medium_id in medium_ids]


def material_label(medium: dict[str, Any]) -> str:
    medium_id = str(medium["id"])
    if medium_id == "sandstone_medium":
        return "sandstone"
    return medium_id


def build_cases(
    media: list[dict[str, Any]],
    *,
    temperatures: list[float],
    pressures: list[float],
    time_points_ms: list[float],
    frequency_hz: float,
    seed: int,
) -> list[ExperimentCase]:
    randomizer = random.Random(seed)
    records: list[ExperimentCase] = []
    serial = 1

    for medium in media:
        material = material_label(medium)
        for temperature_c in temperatures:
            for pressure_mpa in pressures:
                for time_ms in time_points_ms:
                    source = dict(DEFAULT_SOURCE)
                    source["frequency_hz"] = frequency_hz
                    case = ExperimentCase(
                        case_id=f"case_{serial:03d}_{material}",
                        material=material,
                        medium_id=str(medium["id"]),
                        temperature_c=float(temperature_c),
                        pressure_mpa=float(pressure_mpa),
                        time_ms=float(time_ms),
                        frequency_hz=float(frequency_hz),
                        boundary_conditions=dict(DEFAULT_BOUNDARY_CONDITIONS),
                        input={
                            "scenario": {
                                "temperature_c": float(temperature_c),
                                "pressure_mpa": float(pressure_mpa),
                                "time_ms": float(time_ms),
                            },
                            "source": source,
                            "probe": dict(DEFAULT_PROBE),
                            "domain": json.loads(json.dumps(DEFAULT_DOMAIN)),
                        },
                    )
                    records.append(case)
                    serial += 1

    randomizer.shuffle(records)
    return records


def trim_cases(cases: list[ExperimentCase], num_cases: int | None) -> list[ExperimentCase]:
    if num_cases is None:
        return cases
    if num_cases <= 0:
        raise ValueError("--num-cases must be greater than zero.")
    if num_cases > len(cases):
        raise ValueError(
            f"Requested {num_cases} cases but only {len(cases)} are available with the current grid."
        )
    trimmed = cases[:num_cases]
    renumbered: list[ExperimentCase] = []
    for index, case in enumerate(trimmed, start=1):
        suffix = case.case_id.split("_", 2)[-1]
        renumbered.append(
            ExperimentCase(
                case_id=f"case_{index:03d}_{suffix}",
                material=case.material,
                medium_id=case.medium_id,
                temperature_c=case.temperature_c,
                pressure_mpa=case.pressure_mpa,
                time_ms=case.time_ms,
                frequency_hz=case.frequency_hz,
                boundary_conditions=case.boundary_conditions,
                input=case.input,
            )
        )
    return renumbered


def write_cases(output_path: Path, cases: list[ExperimentCase]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")


def write_metadata(
    output_path: Path,
    *,
    catalog_path: Path,
    cases: list[ExperimentCase],
    seed: int,
    materials: list[str],
    temperatures: list[float],
    pressures: list[float],
    time_points_ms: list[float],
    frequency_hz: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generator": "scripts/generate_experiment_inputs.py",
        "catalog_path": str(catalog_path),
        "case_count": len(cases),
        "seed": seed,
        "materials": materials,
        "temperatures_c": temperatures,
        "pressures_mpa": pressures,
        "time_ms": time_points_ms,
        "frequency_hz": frequency_hz,
        "boundary_conditions": dict(DEFAULT_BOUNDARY_CONDITIONS),
        "domain": json.loads(json.dumps(DEFAULT_DOMAIN)),
        "source_template": {
            **DEFAULT_SOURCE,
            "frequency_hz": frequency_hz,
        },
        "probe_template": dict(DEFAULT_PROBE),
        "case_ids": [case.case_id for case in cases],
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    metadata_output = args.metadata_output
    if metadata_output == DEFAULT_METADATA_OUTPUT and args.output != DEFAULT_OUTPUT:
        metadata_output = args.output.with_suffix(".metadata.json")
    catalog = load_catalog(args.catalog)
    media = resolve_media(catalog, list(args.materials))
    cases = build_cases(
        media,
        temperatures=list(args.temperatures),
        pressures=list(args.pressures),
        time_points_ms=[float(value) for value in args.time_ms],
        frequency_hz=float(args.frequency_hz),
        seed=int(args.seed),
    )
    cases = trim_cases(cases, args.num_cases)
    write_cases(args.output, cases)
    write_metadata(
        metadata_output,
        catalog_path=args.catalog,
        cases=cases,
        seed=int(args.seed),
        materials=[material_label(medium) for medium in media],
        temperatures=[float(value) for value in args.temperatures],
        pressures=[float(value) for value in args.pressures],
        time_points_ms=[float(value) for value in args.time_ms],
        frequency_hz=float(args.frequency_hz),
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "case_count": len(cases),
                "output": str(args.output),
                "metadata_output": str(metadata_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
