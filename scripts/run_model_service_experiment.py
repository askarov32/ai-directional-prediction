#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("artifacts/data_experiments/inputs/model_comparison_inputs.jsonl")
DEFAULT_OUTPUT_DIR = Path("artifacts/data_experiments/results")
DEFAULT_CATALOG = Path("backend/data/media/catalog.json")

SERVICE_CONFIG: dict[str, dict[str, str]] = {
    "pinn": {
        "local_url": f"http://localhost:{os.getenv('PINN_SERVICE_PORT', '9003')}",
        "compose_url": "http://pinn-service:9000",
        "representation": "physics_informed",
        "routing_hint": "pinn",
    },
    "mgn": {
        "local_url": f"http://localhost:{os.getenv('MGN_SERVICE_PORT', '9001')}",
        "compose_url": "http://mgn-service:9000",
        "representation": "graph",
        "routing_hint": "meshgraphnet",
    },
    "fno": {
        "local_url": f"http://localhost:{os.getenv('FNO_SERVICE_PORT', '9002')}",
        "compose_url": "http://fno-service:9000",
        "representation": "grid",
        "routing_hint": "fno",
    },
    "transformer": {
        "local_url": f"http://localhost:{os.getenv('TRANSFORMER_SERVICE_PORT', '9004')}",
        "compose_url": "http://transformer-service:9000",
        "representation": "tokenset",
        "routing_hint": "transformer",
    },
}

MODEL_DOMAIN_POLICY: dict[str, dict[str, Any]] = {
    "pinn": {
        "supported_domain_types": ["rect_2d", "rect_3d"],
        "default_domain_type": "rect_3d",
        "allow_3d": True,
    },
    "mgn": {
        "supported_domain_types": ["rect_2d", "rect_3d"],
        "default_domain_type": "rect_3d",
        "allow_3d": True,
    },
    "transformer": {
        "supported_domain_types": ["rect_2d", "rect_3d"],
        "default_domain_type": "rect_3d",
        "allow_3d": True,
    },
    "fno": {
        "supported_domain_types": ["rect_2d"],
        "default_domain_type": "rect_2d",
        "allow_3d": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the same canonical experiment cases through four direct model-service /predict endpoints."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["pinn", "mgn", "fno", "transformer"],
        choices=sorted(SERVICE_CONFIG.keys()),
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip one-time /health and /ready checks before the experiment run.",
    )
    parser.add_argument(
        "--no-clean-output",
        action="store_true",
        help="Keep existing files in output-dir instead of removing previous generated results first.",
    )
    parser.add_argument(
        "--pinn-url",
        default=os.getenv("PINN_EXPERIMENT_URL", SERVICE_CONFIG["pinn"]["local_url"]),
        help="Base URL for direct PINN calls.",
    )
    parser.add_argument(
        "--mgn-url",
        default=os.getenv("MGN_EXPERIMENT_URL", SERVICE_CONFIG["mgn"]["local_url"]),
        help="Base URL for direct MeshGraphNet calls.",
    )
    parser.add_argument(
        "--fno-url",
        default=os.getenv("FNO_EXPERIMENT_URL", SERVICE_CONFIG["fno"]["local_url"]),
        help="Base URL for direct FNO calls.",
    )
    parser.add_argument(
        "--transformer-url",
        default=os.getenv("TRANSFORMER_EXPERIMENT_URL", SERVICE_CONFIG["transformer"]["local_url"]),
        help="Base URL for direct Transformer calls.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_media(catalog: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {entry["id"]: entry for entry in catalog if isinstance(entry, dict) and "id" in entry}


def adapt_case_for_model(model: str, case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    adapted_case = json.loads(json.dumps(case))
    request_domain = adapted_case.get("input", {}).get("domain", {})
    requested_domain_type = str(request_domain.get("type", "rect_2d"))
    effective_domain_type = requested_domain_type
    adaptation = "none"
    policy = MODEL_DOMAIN_POLICY[model]

    if requested_domain_type not in policy["supported_domain_types"]:
        effective_domain_type = str(policy["default_domain_type"])
        adaptation = f"{requested_domain_type}_to_{effective_domain_type}"
        if effective_domain_type == "rect_2d":
            adapted_case["input"]["domain"]["type"] = "rect_2d"
            adapted_case["input"]["domain"]["size"]["lz"] = 0.0
            adapted_case["input"]["domain"]["resolution"]["nz"] = 1
            adapted_case["input"]["domain"]["boundary_conditions"]["front"] = None
            adapted_case["input"]["domain"]["boundary_conditions"]["back"] = None
            adapted_case["input"]["source"]["z"] = 0.0
            adapted_case["input"]["probe"]["z"] = 0.0
            if len(adapted_case["input"]["source"]["direction"]) == 3:
                adapted_case["input"]["source"]["direction"][2] = 0.0

    meta = {
        "requested_domain_type": requested_domain_type,
        "effective_domain_type": effective_domain_type,
        "domain_adaptation": adaptation,
    }
    return adapted_case, meta


def build_payload(model: str, case: dict[str, Any], medium: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    adapted_case, adaptation_meta = adapt_case_for_model(model, case)
    service = SERVICE_CONFIG[model]
    payload = {
        "medium": medium,
        "scenario": adapted_case["input"]["scenario"],
        "source": adapted_case["input"]["source"],
        "probe": adapted_case["input"]["probe"],
        "domain": adapted_case["input"]["domain"],
        "representation": service["representation"],
        "routing_hint": service["routing_hint"],
    }
    if model == "fno":
        payload["requested_outputs"] = ["direction", "field_summary"]
        payload["grid_policy"] = "service_default"
    return payload, adaptation_meta


def http_get_json(url: str, timeout_seconds: float) -> tuple[int | None, dict[str, Any] | None, str | None]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
            return response.getcode(), json.loads(body), None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8") if exc.fp is not None else ""
        try:
            return exc.code, json.loads(body), None
        except json.JSONDecodeError:
            return exc.code, None, body or str(exc)
    except Exception as exc:  # noqa: BLE001
        return None, None, str(exc)


def http_post_json(
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> tuple[int | None, dict[str, Any] | None, str | None]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
            return response.getcode(), json.loads(body), None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8") if exc.fp is not None else ""
        try:
            return exc.code, json.loads(body), None
        except json.JSONDecodeError:
            return exc.code, None, body or str(exc)
    except Exception as exc:  # noqa: BLE001
        return None, None, str(exc)


def runtime_urls(args: argparse.Namespace) -> dict[str, str]:
    return {
        "pinn": args.pinn_url,
        "mgn": args.mgn_url,
        "fno": args.fno_url,
        "transformer": args.transformer_url,
    }


def preflight(models: list[str], timeout_seconds: float, urls: dict[str, str]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for model in models:
        base_url = urls[model]
        health_status, health_payload, health_error = http_get_json(
            f"{base_url}/health",
            timeout_seconds,
        )
        ready_status, ready_payload, ready_error = http_get_json(
            f"{base_url}/ready",
            timeout_seconds,
        )
        results[model] = {
            "local_url": base_url,
            "compose_url": SERVICE_CONFIG[model]["compose_url"],
            "health": {
                "http_status": health_status,
                "payload": health_payload,
                "error": health_error,
            },
            "ready": {
                "http_status": ready_status,
                "payload": ready_payload,
                "error": ready_error,
            },
        }
    return results


def normalize_success(
    *,
    model: str,
    case: dict[str, Any],
    raw_response: dict[str, Any],
    adaptation_meta: dict[str, Any],
) -> dict[str, Any]:
    if model == "fno":
        prediction = raw_response.get("prediction", {})
        field_summary = raw_response.get("field_summary", {})
        diagnostics = raw_response.get("diagnostics", {})
        service_mode = diagnostics.get("mode", "unknown")
        fallback_used = service_mode == "fallback"
    else:
        prediction = raw_response
        field_summary = raw_response
        diagnostics = raw_response.get("diagnostics", {})
        model_version = str(raw_response.get("model_version", ""))
        fallback_reason = raw_response.get("extra_metrics", {}).get("fallback_reason")
        if model == "mgn":
            fallback_used = "fallback" in model_version or fallback_reason is not None
            service_mode = "fallback" if fallback_used else "rollout"
        else:
            fallback_used = False
            service_mode = diagnostics.get("mode", "checkpoint")

    return {
        "case_id": case["case_id"],
        "model": model,
        "status": "ok",
        "service_mode": service_mode,
        "fallback_used": fallback_used,
        "requested_domain_type": adaptation_meta["requested_domain_type"],
        "effective_domain_type": adaptation_meta["effective_domain_type"],
        "domain_adaptation": adaptation_meta["domain_adaptation"],
        "material": case["material"],
        "medium_id": case["medium_id"],
        "temperature_c": case["temperature_c"],
        "pressure_mpa": case["pressure_mpa"],
        "time_ms": case["time_ms"],
        "frequency_hz": case["frequency_hz"],
        "source_x": case["input"]["source"]["x"],
        "source_y": case["input"]["source"]["y"],
        "source_z": case["input"]["source"]["z"],
        "probe_x": case["input"]["probe"]["x"],
        "probe_y": case["input"]["probe"]["y"],
        "probe_z": case["input"]["probe"]["z"],
        "direction_x": _component(prediction.get("direction_vector"), 0),
        "direction_y": _component(prediction.get("direction_vector"), 1),
        "direction_z": _component(prediction.get("direction_vector"), 2),
        "azimuth_deg": prediction.get("azimuth_deg"),
        "elevation_deg": prediction.get("elevation_deg"),
        "magnitude": prediction.get("magnitude"),
        "travel_time_ms_pred": prediction.get("travel_time_ms"),
        "max_displacement": field_summary.get("max_displacement"),
        "max_temperature_perturbation": field_summary.get("max_temperature_perturbation"),
        "wave_type": prediction.get("wave_type"),
        "model_version": raw_response.get("model_version"),
        "error_code": "",
        "error_message": "",
        "http_status": 200,
    }


def normalize_error(
    *,
    model: str,
    case: dict[str, Any],
    http_status: int | None,
    body: dict[str, Any] | None,
    error_text: str | None,
    adaptation_meta: dict[str, Any],
) -> dict[str, Any]:
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict):
        error_code = detail.get("code") or body.get("code") or "HTTP_ERROR"
        error_message = detail.get("message") or detail.get("detail") or json.dumps(detail)
    elif isinstance(body, dict):
        error_code = body.get("code") or "HTTP_ERROR"
        error_message = body.get("message") or body.get("detail") or json.dumps(body)
    else:
        error_code = "REQUEST_FAILED"
        error_message = error_text or "Unknown request failure"
    return {
        "case_id": case["case_id"],
        "model": model,
        "status": "error",
        "service_mode": "error",
        "fallback_used": False,
        "requested_domain_type": adaptation_meta["requested_domain_type"],
        "effective_domain_type": adaptation_meta["effective_domain_type"],
        "domain_adaptation": adaptation_meta["domain_adaptation"],
        "material": case["material"],
        "medium_id": case["medium_id"],
        "temperature_c": case["temperature_c"],
        "pressure_mpa": case["pressure_mpa"],
        "time_ms": case["time_ms"],
        "frequency_hz": case["frequency_hz"],
        "source_x": case["input"]["source"]["x"],
        "source_y": case["input"]["source"]["y"],
        "source_z": case["input"]["source"]["z"],
        "probe_x": case["input"]["probe"]["x"],
        "probe_y": case["input"]["probe"]["y"],
        "probe_z": case["input"]["probe"]["z"],
        "direction_x": "",
        "direction_y": "",
        "direction_z": "",
        "azimuth_deg": "",
        "elevation_deg": "",
        "magnitude": "",
        "travel_time_ms_pred": "",
        "max_displacement": "",
        "max_temperature_perturbation": "",
        "wave_type": "",
        "model_version": "",
        "error_code": error_code,
        "error_message": error_message,
        "http_status": http_status or "",
    }


def _component(values: Any, index: int) -> float | str:
    if not isinstance(values, list) or len(values) <= index:
        return ""
    return values[index]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "model",
        "status",
        "service_mode",
        "fallback_used",
        "requested_domain_type",
        "effective_domain_type",
        "domain_adaptation",
        "material",
        "medium_id",
        "temperature_c",
        "pressure_mpa",
        "time_ms",
        "frequency_hz",
        "source_x",
        "source_y",
        "source_z",
        "probe_x",
        "probe_y",
        "probe_z",
        "direction_x",
        "direction_y",
        "direction_z",
        "azimuth_deg",
        "elevation_deg",
        "magnitude",
        "travel_time_ms_pred",
        "max_displacement",
        "max_temperature_perturbation",
        "wave_type",
        "model_version",
        "error_code",
        "error_message",
        "http_status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def prepare_output_dir(output_dir: Path, *, clean_output: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not clean_output:
        return
    for child in ("raw", "normalized"):
        path = output_dir / child
        if path.exists():
            shutil.rmtree(path)
    for child in ("summary.csv", "summary.json", "run_metadata.json"):
        path = output_dir / child
        if path.exists():
            path.unlink()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    run_id = str(uuid.uuid4())
    prepare_output_dir(args.output_dir, clean_output=not args.no_clean_output)
    catalog = load_json(args.catalog)
    media = resolve_media(catalog)
    cases = load_jsonl(args.input)

    raw_dir = args.output_dir / "raw"
    normalized_dir = args.output_dir / "normalized"
    raw_requests: list[dict[str, Any]] = []
    raw_responses: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    urls = runtime_urls(args)

    preflight_report = {}
    if not args.skip_preflight:
        preflight_report = preflight(list(args.models), float(args.timeout_seconds), urls)
    write_json(
        args.output_dir / "run_metadata.json",
        {
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "input_path": str(args.input),
            "output_dir": str(args.output_dir),
            "models": list(args.models),
            "urls": urls,
            "timeout_seconds": float(args.timeout_seconds),
            "preflight": preflight_report,
        },
    )

    for case in cases:
        medium_id = case["medium_id"]
        medium = media.get(medium_id)
        if medium is None:
            for model in args.models:
                summary_rows.append(
                    normalize_error(
                        model=model,
                        case=case,
                        http_status=None,
                        body={"code": "UNKNOWN_MEDIUM", "message": f"Medium not found: {medium_id}"},
                        error_text=None,
                        adaptation_meta={
                            "requested_domain_type": case.get("requested_domain_type", case.get("input", {}).get("domain", {}).get("type", "")),
                            "effective_domain_type": case.get("requested_domain_type", case.get("input", {}).get("domain", {}).get("type", "")),
                            "domain_adaptation": "none",
                        },
                    )
                )
            continue

        for model in args.models:
            payload, adaptation_meta = build_payload(model, case, medium)
            predict_url = f"{urls[model]}/predict"
            raw_requests.append(
                {
                    "case_id": case["case_id"],
                    "model": model,
                    "url": predict_url,
                    "adaptation_meta": adaptation_meta,
                    "payload": payload,
                }
            )
            http_status, body, error_text = http_post_json(
                predict_url,
                payload,
                float(args.timeout_seconds),
            )
            raw_responses.append(
                {
                    "case_id": case["case_id"],
                    "model": model,
                    "url": predict_url,
                    "http_status": http_status,
                    "adaptation_meta": adaptation_meta,
                    "response": body,
                    "error_text": error_text,
                }
            )
            if http_status is not None and 200 <= http_status < 300 and isinstance(body, dict):
                normalized = normalize_success(
                    model=model,
                    case=case,
                    raw_response=body,
                    adaptation_meta=adaptation_meta,
                )
                normalized["http_status"] = http_status
            else:
                normalized = normalize_error(
                    model=model,
                    case=case,
                    http_status=http_status,
                    body=body,
                    error_text=error_text,
                    adaptation_meta=adaptation_meta,
                )
            summary_rows.append(normalized)

    write_jsonl(raw_dir / "requests.jsonl", raw_requests)
    write_jsonl(raw_dir / "responses.jsonl", raw_responses)
    write_jsonl(normalized_dir / "results.jsonl", summary_rows)
    write_summary_csv(args.output_dir / "summary.csv", summary_rows)

    ok_count = sum(1 for row in summary_rows if row["status"] == "ok")
    fallback_count = sum(1 for row in summary_rows if row["status"] == "ok" and row["fallback_used"])
    error_count = sum(1 for row in summary_rows if row["status"] == "error")
    summary_payload = {
        "status": "ok",
        "run_id": run_id,
        "input_path": str(args.input),
        "output_dir": str(args.output_dir),
        "case_count": len(cases),
        "models": list(args.models),
        "predict_call_count": len(cases) * len(args.models),
        "ok_count": ok_count,
        "fallback_count": fallback_count,
        "error_count": error_count,
        "elapsed_seconds": round(time.time() - started_at, 3),
        "preflight": preflight_report,
    }
    write_json(args.output_dir / "summary.json", summary_payload)
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
