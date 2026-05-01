from __future__ import annotations

import argparse
import copy
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = BASE_DIR / "inputs" / "granite_scenarios.json"
DEFAULT_OUTPUT_PATH = BASE_DIR / "outputs" / "granite_predictions.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated prediction analytics scenarios against the thesis backend."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Scenario JSON file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Predictions output JSON file.")
    parser.add_argument(
        "--backend-url",
        default=None,
        help="Backend API base URL, for example http://127.0.0.1:8000/api/v1",
    )
    parser.add_argument(
        "--medium-id",
        default=None,
        help="Medium id override. Defaults to the value in the input JSON (granite).",
    )
    parser.add_argument("--model", default=None, help="Model id override. Defaults to the value in the input JSON.")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def http_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url=url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc


def build_request(
    baseline_request: dict[str, Any],
    *,
    medium_id: str,
    model: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "medium_id": medium_id,
        "scenario": {},
        "source": {},
        "probe": {},
        "domain": {},
    }
    deep_merge(payload, baseline_request)
    if overrides:
        deep_merge(payload, overrides)
    return payload


def fetch_medium(api_base: str, medium_id: str, timeout: float) -> dict[str, Any]:
    return http_json(f"{api_base}/media/{medium_id}", timeout=timeout)


def post_prediction(api_base: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    return http_json(f"{api_base}/predictions", method="POST", payload=payload, timeout=timeout)


def build_comparison_cases(
    config: dict[str, Any],
    *,
    medium_id: str,
    model: str,
) -> list[dict[str, Any]]:
    baseline_request = config["baseline_request"]
    cases = []
    for item in config.get("comparison_cases", []):
        payload = build_request(
            baseline_request,
            medium_id=medium_id,
            model=model,
            overrides=item.get("overrides"),
        )
        cases.append(
            {
                "group": "comparison_cases",
                "scenario_id": item["id"],
                "label": item["label"],
                "notes": item.get("notes", ""),
                "variables": {},
                "request": payload,
            }
        )
    return cases


def build_sweep_records(
    config: dict[str, Any],
    *,
    medium_id: str,
    model: str,
) -> list[dict[str, Any]]:
    baseline_request = config["baseline_request"]
    sweeps = config["sweeps"]
    records: list[dict[str, Any]] = []

    amplitude_sweep = sweeps["amplitude_time_series"]
    for amplitude in amplitude_sweep["amplitudes"]:
        for time_ms in amplitude_sweep["time_values_ms"]:
            payload = build_request(
                baseline_request,
                medium_id=medium_id,
                model=model,
                overrides=deep_merge(
                    copy.deepcopy(amplitude_sweep["fixed_overrides"]),
                    {"scenario": {"time_ms": time_ms}, "source": {"amplitude": amplitude}},
                ),
            )
            records.append(
                {
                    "group": "amplitude_time_series",
                    "scenario_id": f"amp_{amplitude:g}_time_{time_ms:g}",
                    "label": f"Amplitude {amplitude:g}, t={time_ms:g} ms",
                    "variables": {"amplitude": amplitude, "time_ms": time_ms},
                    "request": payload,
                }
            )

    temperature_sweep = sweeps["temperature_sensitivity"]
    for temperature_c in temperature_sweep["temperatures_c"]:
        payload = build_request(
            baseline_request,
            medium_id=medium_id,
            model=model,
            overrides=deep_merge(
                copy.deepcopy(temperature_sweep["fixed_overrides"]),
                {"scenario": {"temperature_c": temperature_c}},
            ),
        )
        records.append(
            {
                "group": "temperature_sensitivity",
                "scenario_id": f"temp_{temperature_c:g}",
                "label": f"Temperature {temperature_c:g} C",
                "variables": {"temperature_c": temperature_c},
                "request": payload,
            }
        )

    pressure_sweep = sweeps["pressure_sensitivity"]
    for pressure_mpa in pressure_sweep["pressures_mpa"]:
        payload = build_request(
            baseline_request,
            medium_id=medium_id,
            model=model,
            overrides=deep_merge(
                copy.deepcopy(pressure_sweep["fixed_overrides"]),
                {"scenario": {"pressure_mpa": pressure_mpa}},
            ),
        )
        records.append(
            {
                "group": "pressure_sensitivity",
                "scenario_id": f"pressure_{pressure_mpa:g}",
                "label": f"Pressure {pressure_mpa:g} MPa",
                "variables": {"pressure_mpa": pressure_mpa},
                "request": payload,
            }
        )

    frequency_sweep = sweeps["frequency_sensitivity"]
    for frequency_hz in frequency_sweep["frequencies_hz"]:
        payload = build_request(
            baseline_request,
            medium_id=medium_id,
            model=model,
            overrides=deep_merge(
                copy.deepcopy(frequency_sweep["fixed_overrides"]),
                {"source": {"frequency_hz": frequency_hz}},
            ),
        )
        records.append(
            {
                "group": "frequency_sensitivity",
                "scenario_id": f"frequency_{frequency_hz:g}",
                "label": f"Frequency {frequency_hz:g} Hz",
                "variables": {"frequency_hz": frequency_hz},
                "request": payload,
            }
        )

    heatmap_sweep = sweeps["temperature_pressure_heatmap"]
    for temperature_c in heatmap_sweep["temperatures_c"]:
        for pressure_mpa in heatmap_sweep["pressures_mpa"]:
            payload = build_request(
                baseline_request,
                medium_id=medium_id,
                model=model,
                overrides=deep_merge(
                    copy.deepcopy(heatmap_sweep["fixed_overrides"]),
                    {"scenario": {"temperature_c": temperature_c, "pressure_mpa": pressure_mpa}},
                ),
            )
            records.append(
                {
                    "group": "temperature_pressure_heatmap",
                    "scenario_id": f"temp_{temperature_c:g}_pressure_{pressure_mpa:g}",
                    "label": f"Temp {temperature_c:g} C / Pressure {pressure_mpa:g} MPa",
                    "variables": {"temperature_c": temperature_c, "pressure_mpa": pressure_mpa},
                    "request": payload,
                }
            )

    surface_sweep = sweeps["three_d_surface"]
    for probe_z in surface_sweep["probe_z_values"]:
        for time_ms in surface_sweep["time_values_ms"]:
            payload = build_request(
                baseline_request,
                medium_id=medium_id,
                model=model,
                overrides=deep_merge(
                    copy.deepcopy(surface_sweep["fixed_overrides"]),
                    {"scenario": {"time_ms": time_ms}, "probe": {"z": probe_z}},
                ),
            )
            records.append(
                {
                    "group": "three_d_surface",
                    "scenario_id": f"probe_z_{probe_z:g}_time_{time_ms:g}",
                    "label": f"Probe z={probe_z:g}, t={time_ms:g} ms",
                    "variables": {"probe_z": probe_z, "time_ms": time_ms},
                    "request": payload,
                }
            )

    return records


def run_predictions(
    records: list[dict[str, Any]],
    *,
    api_base: str,
    timeout: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    completed: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        print(
            f"[{index:03d}/{len(records):03d}] {record['group']} :: {record['scenario_id']}",
            file=sys.stderr,
        )
        try:
            response = post_prediction(api_base, record["request"], timeout)
            completed.append(
                {
                    "group": record["group"],
                    "scenario_id": record["scenario_id"],
                    "label": record["label"],
                    "notes": record.get("notes", ""),
                    "variables": record["variables"],
                    "request": record["request"],
                    "response": response,
                }
            )
        except Exception as exc:  # noqa: BLE001
            error_payload = {
                "group": record["group"],
                "scenario_id": record["scenario_id"],
                "label": record["label"],
                "variables": record["variables"],
                "request": record["request"],
                "error": str(exc),
            }
            errors.append(error_payload)
            print(f"  ! {error_payload['error']}", file=sys.stderr)
    return completed, errors


def group_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["group"], []).append(record)
    return grouped


def main() -> int:
    args = parse_args()
    config = load_json(args.input)
    api_base = (args.backend_url or config.get("backend_api_base") or "").rstrip("/")
    if not api_base:
        raise RuntimeError("Backend API base URL is not configured.")

    medium_id = args.medium_id or config["default_medium_id"]
    model = args.model or config["default_model"]

    medium = fetch_medium(api_base, medium_id, args.timeout)
    comparison_cases = build_comparison_cases(config, medium_id=medium_id, model=model)
    sweep_records = build_sweep_records(config, medium_id=medium_id, model=model)
    all_records = comparison_cases + sweep_records
    completed, errors = run_predictions(all_records, api_base=api_base, timeout=args.timeout)

    output_payload = {
        "title": config.get("title", "Prediction analytics"),
        "description": config.get("description", ""),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend_api_base": api_base,
        "medium_id": medium_id,
        "model": model,
        "medium": medium,
        "counts": {
            "requested": len(all_records),
            "completed": len(completed),
            "errors": len(errors),
        },
        "results": group_records(completed),
        "errors": errors,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(output_payload, file, indent=2)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "medium_id": medium_id,
                "model": model,
                "requested": len(all_records),
                "completed": len(completed),
                "errors": len(errors),
            },
            indent=2,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
