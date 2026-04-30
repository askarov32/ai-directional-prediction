from __future__ import annotations

import torch
from torch import nn


class MLP_PINN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 192,
        depth: int = 6,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        activation_layer = _resolve_activation(activation)

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), activation_layer()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), activation_layer()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


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
