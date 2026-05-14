from __future__ import annotations

import pytest
import torch

from fno_service.models import FNO2d, SpectralConv2d


def test_spectral_conv_2d_preserves_spatial_shape() -> None:
    layer = SpectralConv2d(in_channels=3, out_channels=5, modes_x=4, modes_y=4)
    inputs = torch.randn(2, 3, 16, 12)

    outputs = layer(inputs)

    assert outputs.shape == (2, 5, 16, 12)
    assert torch.isfinite(outputs).all()


def test_fno_2d_forward_shape_and_finite_values() -> None:
    model = FNO2d(in_channels=11, out_channels=4, width=16, modes_x=4, modes_y=4, depth=3)
    inputs = torch.randn(2, 11, 16, 12)

    outputs = model(inputs)

    assert outputs.shape == (2, 4, 16, 12)
    assert torch.isfinite(outputs).all()


def test_fno_2d_rejects_invalid_input_rank() -> None:
    model = FNO2d(in_channels=11, out_channels=4, width=8, modes_x=2, modes_y=2, depth=1)

    with pytest.raises(ValueError, match="input shape"):
        model(torch.randn(1, 11, 1, 8, 8))
