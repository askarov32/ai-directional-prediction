from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F

from fno_service.models.layers import SpectralConv2d


class FNO2d(nn.Module):
    """Minimal 2D Fourier Neural Operator for MVP field prediction."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        width: int = 32,
        modes_x: int = 12,
        modes_y: int = 12,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("FNO2d depth must be positive.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.depth = depth

        self.input_projection = nn.Conv2d(in_channels, width, kernel_size=1)
        self.spectral_layers = nn.ModuleList(
            [SpectralConv2d(width, width, modes_x=modes_x, modes_y=modes_y) for _ in range(depth)]
        )
        self.pointwise_layers = nn.ModuleList([nn.Conv2d(width, width, kernel_size=1) for _ in range(depth)])
        self.output_projection = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width * 2, out_channels, kernel_size=1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4:
            raise ValueError("FNO2d expects input shape [batch, channels, height, width].")
        x = self.input_projection(inputs)
        for spectral, pointwise in zip(self.spectral_layers, self.pointwise_layers, strict=True):
            x = F.gelu(spectral(x) + pointwise(x))
        return self.output_projection(x)
