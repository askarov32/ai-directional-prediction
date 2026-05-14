from __future__ import annotations

import torch
from torch import Tensor, nn


class SpectralConv2d(nn.Module):
    """2D Fourier convolution over the lowest retained frequency modes."""

    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("SpectralConv2d channels must be positive.")
        if modes_x <= 0 or modes_y <= 0:
            raise ValueError("SpectralConv2d modes must be positive.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        scale = 1.0 / (in_channels * out_channels)
        self.weights_pos = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_y, modes_x, dtype=torch.cfloat)
        )
        self.weights_neg = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_y, modes_x, dtype=torch.cfloat)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4:
            raise ValueError("SpectralConv2d expects input shape [batch, channels, height, width].")
        batch, _, height, width = inputs.shape
        x_ft = torch.fft.rfft2(inputs)
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=inputs.device,
        )

        modes_y = min(self.modes_y, height)
        modes_x = min(self.modes_x, width // 2 + 1)
        out_ft[:, :, :modes_y, :modes_x] = _complex_multiply(
            x_ft[:, :, :modes_y, :modes_x],
            self.weights_pos[:, :, :modes_y, :modes_x],
        )
        out_ft[:, :, -modes_y:, :modes_x] = _complex_multiply(
            x_ft[:, :, -modes_y:, :modes_x],
            self.weights_neg[:, :, :modes_y, :modes_x],
        )
        return torch.fft.irfft2(out_ft, s=(height, width))


def _complex_multiply(inputs: Tensor, weights: Tensor) -> Tensor:
    return torch.einsum("bixy,ioxy->boxy", inputs, weights)
