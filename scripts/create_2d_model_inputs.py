#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("artifacts/data_experiments/inputs/model_comparison_inputs.jsonl")
DEFAULT_OUTPUT = Path("artifacts/data_experiments/inputs/model_comparison_inputs_2d.jsonl")
DEFAULT_METADATA_OUTPUT = Path("artifacts/data_experiments/inputs/model_comparison_inputs_2d.metadata.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert existing model comparison experiment cases into clean 2D thesis cases."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument("--domain-nx", type=int, default=128)
    parser.add_argument("--domain-ny", type=int, default=128)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def normalize_xy_direction(direction: Any, source: dict[str, Any], probe: dict[str, Any]) -> list[float]:
    values = list(direction) if isinstance(direction, list) else []
    x = float(values[0]) if len(values) > 0 else 0.0
    y = float(values[1]) if len(values) > 1 else 0.0
    norm = math.hypot(x, y)
    if norm <= 1e-12:
        x = float(probe.get("x", 0.0)) - float(source.get("x", 0.0))
        y = float(probe.get("y", 0.0)) - float(source.get("y", 0.0))
        norm = math.hypot(x, y)
    if norm <= 1e-12:
        return [1.0, 0.0, 0.0]
    return [x / norm, y / norm, 0.0]


def convert_case_to_2d(case: dict[str, Any], *, domain_nx: int, domain_ny: int) -> dict[str, Any]:
    converted = deepcopy(case)
    payload = converted.setdefault("input", {})
    source = payload.setdefault("source", {})
    probe = payload.setdefault("probe", {})
    domain = payload.setdefault("domain", {})
    size = domain.setdefault("size", {})
    resolution = domain.setdefault("resolution", {})
    boundary_conditions = domain.setdefault("boundary_conditions", {})

    converted["requested_domain_type"] = "rect_2d"
    domain["type"] = "rect_2d"
    size["lz"] = 0.0
    resolution["nx"] = int(domain_nx)
    resolution["ny"] = int(domain_ny)
    resolution["nz"] = 1
    boundary_conditions["front"] = None
    boundary_conditions["back"] = None

    source["z"] = 0.0
    probe["z"] = 0.0
    source["direction"] = normalize_xy_direction(source.get("direction"), source, probe)

    return converted


def validate_2d_case(case: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    payload = case.get("input", {})
    source = payload.get("source", {})
    probe = payload.get("probe", {})
    domain = payload.get("domain", {})
    size = domain.get("size", {})
    resolution = domain.get("resolution", {})
    direction = source.get("direction", [])

    if case.get("requested_domain_type") != "rect_2d":
        errors.append("requested_domain_type is not rect_2d")
    if domain.get("type") != "rect_2d":
        errors.append("domain.type is not rect_2d")
    if float(size.get("lz", -1.0)) != 0.0:
        errors.append("domain.size.lz is not 0")
    if int(resolution.get("nz", -1)) != 1:
        errors.append("domain.resolution.nz is not 1")
    if float(source.get("z", -1.0)) != 0.0:
        errors.append("source.z is not 0")
    if float(probe.get("z", -1.0)) != 0.0:
        errors.append("probe.z is not 0")
    if len(direction) != 3:
        errors.append("source.direction does not have 3 components")
    elif abs(float(direction[2])) > 1e-12:
        errors.append("source.direction[2] is not 0")

    return errors


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_metadata(
    path: Path,
    *,
    input_path: Path,
    output_path: Path,
    source_count: int,
    converted_records: list[dict[str, Any]],
    validation_errors: dict[str, list[str]],
) -> None:
    materials = sorted({str(record.get("material", "")) for record in converted_records if record.get("material")})
    payload = {
        "generator": "scripts/create_2d_model_inputs.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "output": str(output_path),
        "source_case_count": source_count,
        "converted_case_count": len(converted_records),
        "materials": materials,
        "domain": {
            "requested_domain_type": "rect_2d",
            "domain_type": "rect_2d",
            "source_z": 0.0,
            "probe_z": 0.0,
            "direction_z": 0.0,
            "domain_lz": 0.0,
            "domain_nz": 1,
        },
        "validation": {
            "valid": not validation_errors,
            "invalid_case_count": len(validation_errors),
            "errors": validation_errors,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input)
    converted = [
        convert_case_to_2d(record, domain_nx=args.domain_nx, domain_ny=args.domain_ny)
        for record in records
    ]
    validation_errors = {
        str(record.get("case_id", index)): errors
        for index, record in enumerate(converted, start=1)
        if (errors := validate_2d_case(record))
    }
    if validation_errors:
        sample = next(iter(validation_errors.items()))
        raise ValueError(f"2D conversion failed for {len(validation_errors)} cases. First error: {sample}")

    write_jsonl(args.output, converted)
    write_metadata(
        args.metadata_output,
        input_path=args.input,
        output_path=args.output,
        source_count=len(records),
        converted_records=converted,
        validation_errors=validation_errors,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "source_case_count": len(records),
                "converted_case_count": len(converted),
                "output": str(args.output),
                "metadata_output": str(args.metadata_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
