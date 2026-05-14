#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_COLUMNS = [
    "case_id",
    "model",
    "status",
    "service_mode",
    "fallback_used",
    "material",
    "temperature_c",
    "pressure_mpa",
    "time_ms",
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
    "http_status",
]

MODEL_ORDER = ["pinn", "mgn", "fno", "transformer"]
MATERIAL_ORDER = ["basalt", "sandstone"]

DISPLACEMENT_SANITY_LIMIT = 1.0
TEMPERATURE_SANITY_LIMIT = 10_000.0
DIRECTION_NORM_TOLERANCE = 0.05


def float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def bool_from_string(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def circular_distance_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def circular_mean_deg(values: list[float]) -> float | None:
    if not values:
        return None
    radians = [math.radians(value) for value in values]
    sin_mean = sum(math.sin(value) for value in radians) / len(radians)
    cos_mean = sum(math.cos(value) for value in radians) / len(radians)
    if math.isclose(sin_mean, 0.0, abs_tol=1e-12) and math.isclose(cos_mean, 0.0, abs_tol=1e-12):
        return 0.0
    return (math.degrees(math.atan2(sin_mean, cos_mean)) + 360.0) % 360.0


def circular_std_deg(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    radians = [math.radians(value) for value in values]
    sin_mean = sum(math.sin(value) for value in radians) / len(radians)
    cos_mean = sum(math.cos(value) for value in radians) / len(radians)
    resultant = math.sqrt(sin_mean**2 + cos_mean**2)
    resultant = min(max(resultant, 1e-12), 1.0)
    return math.degrees(math.sqrt(-2.0 * math.log(resultant)))


def direction_norm(row: dict[str, Any]) -> float | None:
    components = [
        float_or_none(row.get("direction_x")),
        float_or_none(row.get("direction_y")),
        float_or_none(row.get("direction_z")),
    ]
    if any(component is None for component in components):
        return None
    return math.sqrt(sum(float(component) ** 2 for component in components))


def normalize_direction_row(row: dict[str, Any]) -> dict[str, Any]:
    norm = direction_norm(row)
    if norm is None or norm == 0:
        row["direction_norm"] = norm
        row["direction_x_unit"] = None
        row["direction_y_unit"] = None
        row["direction_z_unit"] = None
        return row
    row["direction_norm"] = norm
    row["direction_x_unit"] = float(row["direction_x"]) / norm
    row["direction_y_unit"] = float(row["direction_y"]) / norm
    row["direction_z_unit"] = float(row["direction_z"]) / norm
    return row


def model_sort_key(model: str) -> tuple[int, str]:
    if model in MODEL_ORDER:
        return (MODEL_ORDER.index(model), model)
    return (len(MODEL_ORDER), model)


def material_sort_key(material: str) -> tuple[int, str]:
    if material in MATERIAL_ORDER:
        return (MATERIAL_ORDER.index(material), material)
    return (len(MATERIAL_ORDER), material)


@dataclass
class AnalysisResult:
    rows: list[dict[str, Any]]
    warnings: list[str]
    stats: dict[str, Any]


def validate_columns(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("summary.csv is empty.")
    columns = set(rows[0].keys())
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"summary.csv is missing required columns: {missing}")


def enrich_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for raw_row in rows:
        row = dict(raw_row)
        row["http_status_num"] = int(float(row["http_status"])) if row.get("http_status") not in ("", None) else None
        row["fallback_used_bool"] = bool_from_string(row.get("fallback_used"))
        row["is_checkpoint"] = row.get("service_mode") == "checkpoint" and not row["fallback_used_bool"]
        row["is_fallback"] = row["fallback_used_bool"] or row.get("service_mode") == "fallback"
        row["is_error"] = row.get("status") != "ok" or ((row["http_status_num"] or 0) >= 400)
        row["model_label"] = f"{row['model']} (fallback)" if row["is_fallback"] else row["model"]
        row["max_displacement_value"] = float_or_none(row.get("max_displacement"))
        row["max_temperature_perturbation_value"] = float_or_none(row.get("max_temperature_perturbation"))
        row["travel_time_ms_pred_value"] = float_or_none(row.get("travel_time_ms_pred"))
        row["azimuth_deg_value"] = float_or_none(row.get("azimuth_deg"))
        row["elevation_deg_value"] = float_or_none(row.get("elevation_deg"))
        row["probe_z_value"] = float_or_none(row.get("probe_z"))
        row["source_z_value"] = float_or_none(row.get("source_z"))
        row["is_outlier"] = False
        row["outlier_reasons"] = []
        if row["max_displacement_value"] is not None and row["max_displacement_value"] > DISPLACEMENT_SANITY_LIMIT:
            row["is_outlier"] = True
            row["outlier_reasons"].append("max_displacement")
        if (
            row["max_temperature_perturbation_value"] is not None
            and row["max_temperature_perturbation_value"] > TEMPERATURE_SANITY_LIMIT
        ):
            row["is_outlier"] = True
            row["outlier_reasons"].append("max_temperature_perturbation")
        normalize_direction_row(row)
        enriched.append(row)
    return enriched


def collect_warnings(rows: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    for model, model_rows in sorted(by_model.items(), key=lambda item: model_sort_key(item[0])):
        ok_rows = [row for row in model_rows if not row["is_error"]]
        if ok_rows and all(row["is_fallback"] for row in ok_rows):
            warnings.append(f"{model} has only fallback responses.")

        norms = [row["direction_norm"] for row in ok_rows if row["direction_norm"] is not None]
        if norms and any(abs(norm - 1.0) > DIRECTION_NORM_TOLERANCE for norm in norms):
            warnings.append(f"{model} direction vectors are not consistently unit-normalized.")

        elevations = [row["elevation_deg_value"] for row in ok_rows if row["elevation_deg_value"] is not None]
        if elevations and all(abs(value) < 1e-9 for value in elevations):
            warnings.append(
                f"{model} elevation is always zero. This suggests 2D fallback/adaptation or missing 3D direction output."
            )

        direction_z_values = [float_or_none(row.get("direction_z")) for row in ok_rows if row.get("direction_z") not in ("", None)]
        direction_z_values = [value for value in direction_z_values if value is not None]
        if direction_z_values and all(abs(value) < 1e-9 for value in direction_z_values):
            warnings.append(f"{model} direction_z is always zero.")

        displacement_values = [
            row["max_displacement_value"] for row in ok_rows if row["max_displacement_value"] is not None
        ]
        if displacement_values and max(displacement_values) > DISPLACEMENT_SANITY_LIMIT:
            warnings.append(f"{model} has displacement outliers above sanity limit {DISPLACEMENT_SANITY_LIMIT:g}.")

        temperature_values = [
            row["max_temperature_perturbation_value"]
            for row in ok_rows
            if row["max_temperature_perturbation_value"] is not None
        ]
        if temperature_values and max(temperature_values) > TEMPERATURE_SANITY_LIMIT:
            warnings.append(
                f"{model} has temperature perturbation outliers above sanity limit {TEMPERATURE_SANITY_LIMIT:g}."
            )
    return warnings


def collect_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, Counter[str]] = defaultdict(Counter)
    outliers: list[dict[str, Any]] = []
    for row in rows:
        model = row["model"]
        by_model[model]["total"] += 1
        if row["is_error"]:
            by_model[model]["error"] += 1
            if row["error_code"] == "REQUEST_FAILED" and "timed out" in str(row["error_message"]).lower():
                by_model[model]["timeout"] += 1
        elif row["is_fallback"]:
            by_model[model]["ok_fallback"] += 1
        elif row["is_checkpoint"]:
            by_model[model]["ok_checkpoint"] += 1
        else:
            by_model[model]["ok_other"] += 1

        if row["is_outlier"]:
            outliers.append(
                {
                    "case_id": row["case_id"],
                    "model": row["model"],
                    "material": row["material"],
                    "reasons": list(row["outlier_reasons"]),
                    "max_displacement": row["max_displacement_value"],
                    "max_temperature_perturbation": row["max_temperature_perturbation_value"],
                }
            )

    return {
        "by_model": {
            model: dict(counter)
            for model, counter in sorted(by_model.items(), key=lambda item: model_sort_key(item[0]))
        },
        "outlier_cases": outliers,
        "case_count": len({row["case_id"] for row in rows}),
        "response_count": len(rows),
    }


def filter_rows(
    rows: list[dict[str, Any]],
    *,
    include_fallback: bool,
    only_ok: bool = True,
    exclude_outliers: bool = False,
) -> list[dict[str, Any]]:
    filtered = rows
    if only_ok:
        filtered = [row for row in filtered if not row["is_error"]]
    if not include_fallback:
        filtered = [row for row in filtered if not row["is_fallback"]]
    if exclude_outliers:
        filtered = [row for row in filtered if not row["is_outlier"]]
    return filtered


def load_and_analyze_summary(path: Path) -> AnalysisResult:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    validate_columns(rows)
    enriched = enrich_rows(rows)
    warnings = collect_warnings(enriched)
    stats = collect_stats(enriched)
    return AnalysisResult(rows=enriched, warnings=warnings, stats=stats)

