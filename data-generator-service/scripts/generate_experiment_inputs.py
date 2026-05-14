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
    "z": 0.2,
    "amplitude": 1.0,
    "frequency_hz": 50.0,
    "direction": [1.0, 0.15, 0.1],
}

DEFAULT_PROBE = {
    "x": 0.70,
    "y": 0.55,
    "z": 0.75,
}

SOURCE_VARIANTS = [
    {"x": 0.12, "y": 0.34, "z": 0.18, "direction": [1.0, 0.10, 0.08], "amplitude": 0.8},
    {"x": 0.18, "y": 0.42, "z": 0.26, "direction": [0.92, 0.24, 0.16], "amplitude": 1.0},
    {"x": 0.24, "y": 0.58, "z": 0.34, "direction": [0.85, -0.18, 0.28], "amplitude": 1.2},
    {"x": 0.31, "y": 0.47, "z": 0.22, "direction": [0.78, 0.32, 0.24], "amplitude": 0.9},
]

PROBE_VARIANTS = [
    {"x": 0.68, "y": 0.52, "z": 0.52},
    {"x": 0.74, "y": 0.60, "z": 0.60},
    {"x": 0.58, "y": 0.66, "z": 0.70},
    {"x": 0.80, "y": 0.48, "z": 0.78},
]

BOUNDARY_VARIANTS = [
    {"left": "fixed", "right": "free", "top": "insulated", "bottom": "insulated"},
    {"left": "fixed", "right": "fixed", "top": "free", "bottom": "insulated"},
    {"left": "free", "right": "free", "top": "insulated", "bottom": "fixed"},
]

DEFAULT_DOMAIN = {
    "type": "rect_3d",
    "size": {
        "lx": 1.0,
        "ly": 1.0,
        "lz": 1.0,
    },
    "resolution": {
        "nx": 128,
        "ny": 128,
        "nz": 48,
    },
    "boundary_conditions": {
        **DEFAULT_BOUNDARY_CONDITIONS,
        "front": "insulated",
        "back": "insulated",
    },
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
    requested_domain_type: str
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
        default=[20.0, 60.0, 120.0, 180.0, 260.0, 320.0],
        help="Scenario temperatures in Celsius.",
    )
    parser.add_argument(
        "--pressures",
        nargs="+",
        type=float,
        default=[5.0, 25.0, 60.0],
        help="Scenario pressures in MPa.",
    )
    parser.add_argument(
        "--time-ms",
        nargs="+",
        type=float,
        default=[4.0, 8.0, 12.0, 16.0],
        help="Scenario time points in milliseconds.",
    )
    parser.add_argument(
        "--frequency-hz",
        nargs="+",
        type=float,
        default=[25.0, 50.0, 75.0],
        help="Source frequencies used across generated cases.",
    )
    parser.add_argument(
        "--domain-type",
        choices=["rect_2d", "rect_3d"],
        default="rect_3d",
        help="Requested domain type for generated cases.",
    )
    parser.add_argument("--domain-lx", type=float, default=1.0)
    parser.add_argument("--domain-ly", type=float, default=1.0)
    parser.add_argument("--domain-lz", type=float, default=1.0)
    parser.add_argument("--domain-nx", type=int, default=128)
    parser.add_argument("--domain-ny", type=int, default=128)
    parser.add_argument("--domain-nz", type=int, default=48)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for deterministic ordering.",
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=40,
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
    frequencies_hz: list[float],
    domain_type: str,
    domain_size: dict[str, float],
    domain_resolution: dict[str, int],
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
                    for frequency_hz in frequencies_hz:
                        source_variant = SOURCE_VARIANTS[(serial - 1) % len(SOURCE_VARIANTS)]
                        probe_variant = PROBE_VARIANTS[(serial - 1) % len(PROBE_VARIANTS)]
                        boundary_variant = BOUNDARY_VARIANTS[(serial - 1) % len(BOUNDARY_VARIANTS)]
                        source = dict(DEFAULT_SOURCE)
                        source.update(source_variant)
                        source["frequency_hz"] = frequency_hz
                        probe = dict(DEFAULT_PROBE)
                        probe.update(probe_variant)
                        domain = json.loads(json.dumps(DEFAULT_DOMAIN))
                        domain["type"] = domain_type
                        domain["size"].update(domain_size)
                        domain["resolution"].update(domain_resolution)
                        domain["boundary_conditions"].update(boundary_variant)
                        if domain_type == "rect_2d":
                            source["z"] = 0.0
                            source["direction"] = [source_variant["direction"][0], source_variant["direction"][1], 0.0]
                            probe["z"] = 0.0
                            domain["size"]["lz"] = 0.0
                            domain["resolution"]["nz"] = 1
                            domain["boundary_conditions"]["front"] = None
                            domain["boundary_conditions"]["back"] = None
                        else:
                            max_depth = max(float(domain["size"]["lz"]), 1e-8)
                            source["z"] = min(float(source_variant["z"]), max_depth)
                            probe["z"] = min(float(probe_variant["z"]), max_depth)
                            source["direction"] = list(source_variant["direction"])
                        case = ExperimentCase(
                            case_id=f"case_{serial:03d}_{material}",
                            material=material,
                            medium_id=str(medium["id"]),
                            temperature_c=float(temperature_c),
                            pressure_mpa=float(pressure_mpa),
                            time_ms=float(time_ms),
                            frequency_hz=float(frequency_hz),
                            boundary_conditions=dict(boundary_variant),
                            requested_domain_type=domain_type,
                            input={
                                "scenario": {
                                    "temperature_c": float(temperature_c),
                                    "pressure_mpa": float(pressure_mpa),
                                    "time_ms": float(time_ms),
                                },
                                "source": source,
                                "probe": probe,
                                "domain": domain,
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
                requested_domain_type=case.requested_domain_type,
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
    frequencies_hz: list[float],
    domain_type: str,
    domain_size: dict[str, float],
    domain_resolution: dict[str, int],
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
        "frequency_hz": frequencies_hz,
        "boundary_conditions": BOUNDARY_VARIANTS,
        "domain": {
            "type": domain_type,
            "size": domain_size,
            "resolution": domain_resolution,
        },
        "source_template": {
            **DEFAULT_SOURCE,
            "frequency_hz": frequencies_hz[0] if frequencies_hz else DEFAULT_SOURCE["frequency_hz"],
        },
        "probe_template": dict(DEFAULT_PROBE),
        "source_variants": SOURCE_VARIANTS,
        "probe_variants": PROBE_VARIANTS,
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
        frequencies_hz=[float(value) for value in args.frequency_hz],
        domain_type=args.domain_type,
        domain_size={
            "lx": float(args.domain_lx),
            "ly": float(args.domain_ly),
            "lz": float(args.domain_lz),
        },
        domain_resolution={
            "nx": int(args.domain_nx),
            "ny": int(args.domain_ny),
            "nz": int(args.domain_nz),
        },
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
        frequencies_hz=[float(value) for value in args.frequency_hz],
        domain_type=args.domain_type,
        domain_size={
            "lx": float(args.domain_lx),
            "ly": float(args.domain_ly),
            "lz": float(args.domain_lz),
        },
        domain_resolution={
            "nx": int(args.domain_nx),
            "ny": int(args.domain_ny),
            "nz": int(args.domain_nz),
        },
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
