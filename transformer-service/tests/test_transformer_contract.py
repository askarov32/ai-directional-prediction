from __future__ import annotations

import torch
from pydantic import ValidationError

import pytest

from transformer_service.losses import supervised_mse
from transformer_service.model import OFormer
from transformer_service.service_schemas import TransformerPredictionRequest


def test_oformer_forward_shape():
    model = OFormer(
        input_dim=16,
        query_dim=3,
        output_dim=4,
        d_model=32,
        n_heads=4,
        enc_depth=2,
        dec_depth=2,
        ffn_expansion=2,
        dropout=0.0,
    )
    input_tokens = torch.randn(2, 10, 16)
    query_coords = torch.randn(2, 6, 3)
    out = model(input_tokens, query_coords)
    assert out.shape == (2, 6, 4)


def test_oformer_accepts_unbatched_input():
    model = OFormer(
        input_dim=16,
        query_dim=3,
        output_dim=4,
        d_model=32,
        n_heads=4,
        enc_depth=1,
        dec_depth=1,
        ffn_expansion=2,
        dropout=0.0,
    )
    input_tokens = torch.randn(10, 16)
    query_coords = torch.randn(6, 3)
    out = model(input_tokens, query_coords)
    assert out.shape == (1, 6, 4)


def test_supervised_mse_finite():
    pred = torch.randn(1, 32, 4)
    target = torch.randn(1, 32, 4)
    loss, metrics = supervised_mse(pred, target)
    assert torch.isfinite(loss).item()
    assert metrics["total_loss"] >= 0.0
    for key, value in metrics.items():
        assert value == value  # not NaN


def test_supervised_mse_zero_when_equal():
    pred = torch.ones(1, 8, 4)
    loss, metrics = supervised_mse(pred, pred.clone())
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def _valid_request_payload() -> dict:
    return {
        "medium": {
            "id": "sandstone",
            "name": "Sandstone",
            "category": "sedimentary",
            "properties": {
                "rho": 2200.0,
                "porosity_total": 0.34,
                "porosity_effective": 0.27,
                "vp": 4.0,
                "vs": 2.3,
                "thermal_conductivity": 2.2,
                "heat_capacity": 800.0,
                "thermal_expansion": 1e-5,
            },
        },
        "scenario": {"temperature_c": 20.0, "pressure_mpa": 5.0, "time_ms": 1.0},
        "source": {
            "type": "thermal_pulse",
            "x": 0.1,
            "y": 0.1,
            "z": 0.0,
            "amplitude": 1.0,
            "frequency_hz": 50.0,
            "direction": [1.0, 0.0, 0.0],
        },
        "probe": {"x": 0.5, "y": 0.5, "z": 0.0},
        "domain": {
            "type": "rect_2d",
            "size": {"lx": 1.0, "ly": 1.0, "lz": 0.0},
            "resolution": {"nx": 64, "ny": 64, "nz": 1},
            "boundary_conditions": {
                "left": "fixed",
                "right": "free",
                "top": "insulated",
                "bottom": "insulated",
            },
        },
        "representation": "tokenset",
        "routing_hint": "transformer",
    }


def test_schema_accepts_valid_payload():
    request = TransformerPredictionRequest.model_validate(_valid_request_payload())
    assert request.representation == "tokenset"
    assert request.routing_hint == "transformer"


def test_schema_rejects_bad_representation():
    payload = _valid_request_payload()
    payload["representation"] = "physics_informed"
    with pytest.raises(ValidationError):
        TransformerPredictionRequest.model_validate(payload)


def test_schema_rejects_bad_direction():
    payload = _valid_request_payload()
    payload["source"]["direction"] = [0.0, 0.0, 0.0]
    with pytest.raises(ValidationError):
        TransformerPredictionRequest.model_validate(payload)
