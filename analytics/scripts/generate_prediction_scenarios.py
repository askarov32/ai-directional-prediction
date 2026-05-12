from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODELS = ["meshgraphnet", "fno", "pinn"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate comparable prediction scenarios for all rocks and model types.")
    parser.add_argument("--catalog", default="backend/data/media/catalog.json")
    parser.add_argument("--output", default="analytics/prediction_scenarios/scenarios_all_rocks.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    catalog_path = Path(args.catalog).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    media = json.loads(catalog_path.read_text(encoding="utf-8"))
    scenarios = build_scenarios(media)
    payload = {
        "description": "Comparable thermoelastic prediction scenarios for MeshGraphNet, FNO, and PINN across four geological media.",
        "scenario_count": len(scenarios),
        "models": MODELS,
        "medium_ids": [item["id"] for item in media],
        "scenarios": scenarios,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Scenarios:", output_path)
    print("Scenario count:", len(scenarios))


def build_scenarios(media: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for medium in media:
        medium_id = medium["id"]
        temp_min, temp_max = medium["ranges"]["temperature_c"]
        pressure_min, pressure_max = medium["ranges"]["pressure_mpa"]

        baseline_temp = clamp(120.0, temp_min, temp_max)
        baseline_pressure = clamp(35.0, pressure_min, pressure_max)
        temp_values = [
            clamp(40.0, temp_min, temp_max),
            baseline_temp,
            clamp(min(0.8 * temp_max, temp_max - 20.0), temp_min, temp_max),
        ]
        pressure_values = [
            clamp(10.0, pressure_min, pressure_max),
            baseline_pressure,
            clamp(min(0.5 * pressure_max, pressure_max - 50.0), pressure_min, pressure_max),
        ]
        frequency_values = [25.0, 50.0, 100.0]

        scenario_specs: list[tuple[str, float | str, dict[str, float]]] = [
            ("baseline", "baseline", {"temperature_c": baseline_temp, "pressure_mpa": baseline_pressure, "frequency_hz": 50.0})
        ]
        scenario_specs += [
            ("temperature", value, {"temperature_c": value, "pressure_mpa": baseline_pressure, "frequency_hz": 50.0})
            for value in temp_values
        ]
        scenario_specs += [
            ("pressure", value, {"temperature_c": baseline_temp, "pressure_mpa": value, "frequency_hz": 50.0})
            for value in pressure_values
        ]
        scenario_specs += [
            ("frequency", value, {"temperature_c": baseline_temp, "pressure_mpa": baseline_pressure, "frequency_hz": value})
            for value in frequency_values
        ]
        for temp_value in temp_values:
            for pressure_value in pressure_values:
                scenario_specs.append(
                    (
                        "temperature_pressure",
                        f"{temp_value:g}C_{pressure_value:g}MPa",
                        {"temperature_c": temp_value, "pressure_mpa": pressure_value, "frequency_hz": 50.0},
                    )
                )

        for model in MODELS:
            for index, (sweep_name, sweep_value, values) in enumerate(scenario_specs, start=1):
                scenario_id = f"{medium_id}_{model}_{sweep_name}_{index:03d}"
                scenarios.append(
                    {
                        "id": scenario_id,
                        "model": model,
                        "medium_id": medium_id,
                        "sweep_name": sweep_name,
                        "sweep_value": sweep_value,
                        "request": build_request(
                            model=model,
                            medium_id=medium_id,
                            temperature_c=values["temperature_c"],
                            pressure_mpa=values["pressure_mpa"],
                            frequency_hz=values["frequency_hz"],
                        ),
                    }
                )
    return scenarios


def build_request(*, model: str, medium_id: str, temperature_c: float, pressure_mpa: float, frequency_hz: float) -> dict[str, Any]:
    return {
        "model": model,
        "medium_id": medium_id,
        "scenario": {
            "temperature_c": round(temperature_c, 6),
            "pressure_mpa": round(pressure_mpa, 6),
            "time_ms": 12.0,
        },
        "source": {
            "type": "thermal_pulse",
            "x": 0.15,
            "y": 0.4,
            "z": 0.0,
            "amplitude": 1.0,
            "frequency_hz": round(frequency_hz, 6),
            "direction": [1.0, 0.0, 0.0],
        },
        "probe": {
            "x": 0.7,
            "y": 0.55,
            "z": 0.0,
        },
        "domain": {
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
            "boundary_conditions": {
                "left": "fixed",
                "right": "free",
                "top": "insulated",
                "bottom": "insulated",
            },
        },
    }


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


if __name__ == "__main__":
    main()
