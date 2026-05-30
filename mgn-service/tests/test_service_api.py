from __future__ import annotations

import numpy as np

from service import api


def sample_payload() -> api.PredictionPayload:
    return api.PredictionPayload.model_validate(
        {
            "medium": {
                "id": "sandstone_medium",
                "name": "Sandstone",
                "category": "sedimentary",
                "properties": {"vp": 6.17},
            },
            "scenario": {"temperature_c": 120.0, "pressure_mpa": 35.0, "time_ms": 12.0},
            "source": {
                "type": "thermal_pulse",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "amplitude": 1.0,
                "frequency_hz": 50.0,
                "direction": [1.0, 0.0, 0.0],
            },
            "probe": {"x": 1.0, "y": 1.0, "z": 0.0},
            "domain": {
                "type": "rect_2d",
                "size": {"lx": 1.0, "ly": 1.0, "lz": 0.0},
                "resolution": {"nx": 3, "ny": 2, "nz": 1},
                "boundary_conditions": {
                    "left": "fixed",
                    "right": "free",
                    "top": "insulated",
                    "bottom": "insulated",
                },
            },
            "representation": "graph",
            "routing_hint": "meshgraphnet",
            "requested_outputs": ["field_grid", "field_summary", "diagnostics"],
        }
    )


def test_prediction_payload_accepts_requested_outputs():
    payload = sample_payload()

    assert payload.requested_outputs == ["field_grid", "field_summary", "diagnostics"]


def test_field_grid_is_resampled_from_prediction_bundle(monkeypatch):
    payload = sample_payload()
    final_temperature = np.array([300.0, 310.0, 320.0, 330.0], dtype=np.float32)
    final_u = np.array([0.0, 1.0e-6, 2.0e-6, 3.0e-6], dtype=np.float32)
    final_v = np.array([0.0, 0.0, 1.0e-6, 2.0e-6], dtype=np.float32)
    trajectory = np.stack(
        [
            np.column_stack(
                [
                    final_temperature - 5.0,
                    np.zeros_like(final_u),
                    np.zeros_like(final_v),
                ]
            ),
            np.column_stack([final_temperature, final_u, final_v]),
        ],
        axis=0,
    )
    displacement = np.sqrt(final_u**2 + final_v**2)
    bundle = {
        "trajectory": trajectory,
        "coords": np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "field_names": ["t", "u", "v"],
        "derived": {
            "temperature_change": np.stack(
                [np.zeros_like(final_temperature), final_temperature - 295.0],
                axis=0,
            ),
            "displacement_magnitude": np.stack(
                [np.zeros_like(displacement), displacement],
                axis=0,
            ),
        },
    }
    monkeypatch.setattr(api, "_load_prediction_bundle", lambda: bundle)

    response = api.make_direction_response(payload)
    response = api.enrich_response_with_field_grid(response, payload)
    grid = response["optional_outputs"]["field_grid"]

    assert grid["type"] == "rect_2d"
    assert grid["nx"] == 3
    assert grid["ny"] == 2
    assert "temperature_k" in grid["channels"]
    assert "displacement_magnitude_m" in grid["channels"]
    assert len(grid["channels"]["temperature_k"]["values"]) == 2
    assert len(grid["channels"]["temperature_k"]["values"][0]) == 3
    assert response["optional_outputs"]["field_sources"]["temperature_k"] == "direct_model_output"
    assert response["optional_outputs"]["field_summary"]["max_temperature_k"] == 330.0
    assert response["diagnostics"]["field_grid"]["method"] == "nearest_node_resampling"
