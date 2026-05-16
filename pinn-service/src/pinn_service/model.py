from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn


PINNArchitecture = Literal["mlp", "res_split"]


class MLP_PINN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 192,
        depth: int = 6,
        activation: str = "tanh",
        layer_dims: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        activation_layer = _resolve_activation(activation)
        resolved_layer_dims = _resolve_mlp_layer_dims(hidden_dim=hidden_dim, depth=depth, layer_dims=layer_dims)

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for current_dim in resolved_layer_dims:
            layers.extend([nn.Linear(previous_dim, current_dim), activation_layer()])
            previous_dim = current_dim
        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, activation: str) -> None:
        super().__init__()
        activation_layer = _resolve_activation(activation)
        self.linear1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, width)
        self.activation = activation_layer()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        hidden = self.activation(self.linear1(inputs))
        hidden = self.linear2(hidden)
        return self.activation(hidden + residual)


class FourierCoordinateEncoding(nn.Module):
    def __init__(self, input_dim: int, num_frequencies: int, scale: float) -> None:
        super().__init__()
        if num_frequencies <= 0:
            raise ValueError("fourier_num_frequencies must be positive when Fourier features are enabled.")
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.scale = float(scale)
        frequencies = torch.pow(2.0, torch.arange(num_frequencies, dtype=torch.float32)) * self.scale
        self.register_buffer("frequencies", frequencies, persistent=False)

    @property
    def output_dim(self) -> int:
        return self.input_dim * (1 + 2 * self.num_frequencies)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        expanded = coordinates.unsqueeze(-1) * self.frequencies.view(1, 1, -1) * (2.0 * math.pi)
        sin_features = torch.sin(expanded).reshape(coordinates.shape[0], -1)
        cos_features = torch.cos(expanded).reshape(coordinates.shape[0], -1)
        return torch.cat([coordinates, sin_features, cos_features], dim=1)


class ResSplitPINN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 192,
        num_blocks: int = 4,
        activation: str = "tanh",
        use_fourier_features: bool = False,
        fourier_num_frequencies: int = 6,
        fourier_scale: float = 1.0,
        coord_dim: int = 4,
        material_dim: int = 6,
    ) -> None:
        super().__init__()
        if input_dim != coord_dim + material_dim:
            raise ValueError(
                f"ResSplitPINN expects {coord_dim + material_dim} inputs, received input_dim={input_dim}."
            )
        if output_dim != 4:
            raise ValueError(f"ResSplitPINN expects output_dim=4 for [T, u, v, w], received {output_dim}.")
        if hidden_dim < 64:
            raise ValueError("hidden_dim must be at least 64 for ResSplitPINN.")
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1.")

        activation_layer = _resolve_activation(activation)
        self.coord_dim = coord_dim
        self.material_dim = material_dim
        self.coordinate_encoding: FourierCoordinateEncoding | None = None
        coordinate_input_dim = coord_dim
        if use_fourier_features:
            self.coordinate_encoding = FourierCoordinateEncoding(
                input_dim=coord_dim,
                num_frequencies=fourier_num_frequencies,
                scale=fourier_scale,
            )
            coordinate_input_dim = self.coordinate_encoding.output_dim

        coord_hidden_dim = max(hidden_dim // 2, 64)
        material_hidden_dim = 64

        self.coordinate_encoder = nn.Sequential(
            nn.Linear(coordinate_input_dim, coord_hidden_dim),
            activation_layer(),
        )
        self.material_encoder = nn.Sequential(
            nn.Linear(material_dim, material_hidden_dim),
            activation_layer(),
            nn.Linear(material_hidden_dim, material_hidden_dim),
            activation_layer(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(coord_hidden_dim + material_hidden_dim, hidden_dim),
            activation_layer(),
        )
        self.trunk = nn.Sequential(*[ResidualMLPBlock(hidden_dim, activation) for _ in range(num_blocks)])
        head_hidden_dim = max(hidden_dim // 2, 64)
        self.temperature_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            activation_layer(),
            nn.Linear(head_hidden_dim, 1),
        )
        self.displacement_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            activation_layer(),
            nn.Linear(head_hidden_dim, 3),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        coordinates = inputs[:, : self.coord_dim]
        material = inputs[:, self.coord_dim : self.coord_dim + self.material_dim]
        if self.coordinate_encoding is not None:
            coordinates = self.coordinate_encoding(coordinates)
        coordinate_features = self.coordinate_encoder(coordinates)
        material_features = self.material_encoder(material)
        shared = self.fusion(torch.cat([coordinate_features, material_features], dim=1))
        shared = self.trunk(shared)
        temperature = self.temperature_head(shared)
        displacement = self.displacement_head(shared)
        return torch.cat([temperature, displacement], dim=1)


def create_pinn_model(
    *,
    input_dim: int,
    output_dim: int,
    architecture: PINNArchitecture = "mlp",
    hidden_dim: int = 192,
    depth: int = 6,
    activation: str = "tanh",
    mlp_layer_dims: Sequence[int] | None = None,
    num_blocks: int = 4,
    use_fourier_features: bool = False,
    fourier_num_frequencies: int = 6,
    fourier_scale: float = 1.0,
) -> nn.Module:
    if architecture == "mlp":
        return MLP_PINN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            activation=activation,
            layer_dims=mlp_layer_dims,
        )
    if architecture == "res_split":
        return ResSplitPINN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            activation=activation,
            use_fourier_features=use_fourier_features,
            fourier_num_frequencies=fourier_num_frequencies,
            fourier_scale=fourier_scale,
        )
    raise ValueError(f"Unsupported PINN architecture: {architecture}")


def _resolve_mlp_layer_dims(
    *,
    hidden_dim: int,
    depth: int,
    layer_dims: Sequence[int] | None,
) -> list[int]:
    if layer_dims is not None:
        resolved = [int(value) for value in layer_dims]
        if not resolved:
            raise ValueError("mlp layer_dims must not be empty when provided.")
        if any(value <= 0 for value in resolved):
            raise ValueError("mlp layer_dims must be positive integers.")
        return resolved
    if depth <= 0:
        raise ValueError("depth must be a positive integer.")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be a positive integer.")
    return [hidden_dim for _ in range(depth)]


def parse_layer_dims(raw_value: str | None) -> tuple[int, ...] | None:
    if raw_value is None:
        return None
    parts = [segment.strip() for segment in raw_value.split(",") if segment.strip()]
    if not parts:
        return None
    values = tuple(int(part) for part in parts)
    if any(value <= 0 for value in values):
        raise ValueError("All --mlp-layer-dims values must be positive integers.")
    return values


def _resolve_activation(name: str) -> type[nn.Module]:
    mapping: dict[str, type[nn.Module]] = {
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported activation: {name}")
    return mapping[name]
