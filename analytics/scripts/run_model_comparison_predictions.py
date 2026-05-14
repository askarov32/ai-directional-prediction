from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run model comparison prediction scenarios through the backend API.")
    parser.add_argument("--scenarios", default="analytics/prediction_scenarios/scenarios_all_rocks.json")
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-json", default="analytics/outputs/model_comparison_predictions.json")
    parser.add_argument("--output-csv", default="analytics/outputs/model_comparison_metrics.csv")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of scenarios to execute.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scenario_path = Path(args.scenarios).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    payload = json.loads(scenario_path.read_text(encoding="utf-8"))
    scenarios = payload["scenarios"][: args.limit]
    results = []

    for index, scenario in enumerate(scenarios, start=1):
        print(f"[{index}/{len(scenarios)}] {scenario['id']}")
        results.append(run_prediction(args.backend_url, scenario, args.timeout))

    output = {
        "backend_url": args.backend_url,
        "source_scenarios": str(scenario_path),
        "result_count": len(results),
        "results": results,
    }
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_csv(output_csv, results)
    print("Predictions JSON:", output_json)
    print("Metrics CSV:", output_csv)


def run_prediction(backend_url: str, scenario: dict[str, Any], timeout: float) -> dict[str, Any]:
    url = backend_url.rstrip("/") + "/api/v1/predictions"
    body = json.dumps(scenario["request"]).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
            parsed = json.loads(response_body)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            return {
                "scenario_id": scenario["id"],
                "model": scenario["model"],
                "medium_id": scenario["medium_id"],
                "sweep_name": scenario["sweep_name"],
                "sweep_value": scenario["sweep_value"],
                "status": "ok",
                "latency_ms_client": elapsed_ms,
                "request": scenario["request"],
                "response": parsed,
            }
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        return error_result(scenario, f"HTTP {exc.code}: {error_body}")
    except Exception as exc:  # noqa: BLE001 - CLI report should keep going.
        return error_result(scenario, str(exc))


def error_result(scenario: dict[str, Any], message: str) -> dict[str, Any]:
    return {
        "scenario_id": scenario["id"],
        "model": scenario["model"],
        "medium_id": scenario["medium_id"],
        "sweep_name": scenario["sweep_name"],
        "sweep_value": scenario["sweep_value"],
        "status": "error",
        "error": message,
        "request": scenario["request"],
    }


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "scenario_id",
        "status",
        "model",
        "medium_id",
        "sweep_name",
        "sweep_value",
        "temperature_c",
        "pressure_mpa",
        "frequency_hz",
        "azimuth_deg",
        "elevation_deg",
        "magnitude",
        "travel_time_ms",
        "direction_x",
        "direction_y",
        "direction_z",
        "max_displacement",
        "max_temperature_perturbation",
        "latency_ms_backend",
        "latency_ms_client",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(flatten_result(result))


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    request = result["request"]
    row = {
        "scenario_id": result["scenario_id"],
        "status": result["status"],
        "model": result["model"],
        "medium_id": result["medium_id"],
        "sweep_name": result["sweep_name"],
        "sweep_value": result["sweep_value"],
        "temperature_c": request["scenario"]["temperature_c"],
        "pressure_mpa": request["scenario"]["pressure_mpa"],
        "frequency_hz": request["source"]["frequency_hz"],
        "latency_ms_client": result.get("latency_ms_client"),
        "error": result.get("error"),
    }
    if result["status"] != "ok":
        return row

    response = result["response"]
    prediction = response.get("prediction", {})
    field_summary = response.get("field_summary", {})
    meta = response.get("meta", {})
    direction = prediction.get("direction_vector") or [None, None, None]
    row.update(
        {
            "azimuth_deg": prediction.get("azimuth_deg"),
            "elevation_deg": prediction.get("elevation_deg"),
            "magnitude": prediction.get("magnitude"),
            "travel_time_ms": prediction.get("travel_time_ms"),
            "direction_x": direction[0],
            "direction_y": direction[1],
            "direction_z": direction[2],
            "max_displacement": field_summary.get("max_displacement"),
            "max_temperature_perturbation": field_summary.get("max_temperature_perturbation"),
            "latency_ms_backend": meta.get("latency_ms"),
        }
    )
    return row


if __name__ == "__main__":
    main()
