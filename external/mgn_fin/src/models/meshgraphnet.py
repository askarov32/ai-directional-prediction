"""Conditional MeshGraphNet for thermoelastic wave surrogate modelling."""
from __future__ import annotations

import torch
import torch.nn as nn


def make_mlp(in_dim: int, out_dim: int, hidden_dim: int, n_layers: int, dropout: float = 0.0, layer_norm: bool = True) -> nn.Sequential:
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    dims = [in_dim] + [hidden_dim] * max(0, n_layers - 1) + [out_dim]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            if layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class MessagePassingBlock(nn.Module):
    def __init__(self, latent_dim: int, mlp_layers: int = 3, dropout: float = 0.0, layer_norm: bool = True):
        super().__init__()
        self.edge_mlp = make_mlp(3 * latent_dim, latent_dim, latent_dim, mlp_layers, dropout, layer_norm)
        self.node_mlp = make_mlp(2 * latent_dim, latent_dim, latent_dim, mlp_layers, dropout, layer_norm)
        self.edge_norm = nn.LayerNorm(latent_dim) if layer_norm else nn.Identity()
        self.node_norm = nn.LayerNorm(latent_dim) if layer_norm else nn.Identity()

    def forward(self, h: torch.Tensor, e: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index[0], edge_index[1]

        # Edge update
        e_new = self.edge_mlp(torch.cat([h[src], h[dst], e], dim=-1))
        e_new = self.edge_norm(e_new + e)

        # Aggregation
        # Важно: при AMP/autocast e_new может быть float16,
        # а h может быть float32. scatter_add_ требует одинаковый dtype
        # у agg и src.
        agg = torch.zeros(
            h.shape[0],
            e_new.shape[1],
            device=e_new.device,
            dtype=e_new.dtype,
        )

        index = dst.unsqueeze(1).expand(-1, e_new.shape[1])
        agg.scatter_add_(0, index, e_new)

        # Перед concat с h приводим agg к dtype h
        if agg.dtype != h.dtype:
            agg = agg.to(h.dtype)

        # Node update
        h_new = self.node_mlp(torch.cat([h, agg], dim=-1))
        h_new = self.node_norm(h_new + h)

        return h_new, e_new


class ConditionalMeshGraphNet(nn.Module):
    """Encoder-Processor-Decoder MeshGraphNet.

    The condition is already concatenated into node_features by the data pipeline:
    coords + material + scenario + current dynamic state.
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        out_dim: int,
        latent_dim: int = 128,
        message_passing_steps: int = 10,
        mlp_layers: int = 3,
        dropout: float = 0.05,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        self.message_passing_steps = message_passing_steps

        self.node_encoder = make_mlp(node_in_dim, latent_dim, latent_dim, mlp_layers, dropout, layer_norm)
        self.edge_encoder = make_mlp(edge_in_dim, latent_dim, latent_dim, mlp_layers, dropout, layer_norm)
        self.processor = nn.ModuleList(
            [MessagePassingBlock(latent_dim, mlp_layers, dropout, layer_norm) for _ in range(message_passing_steps)]
        )
        self.decoder = make_mlp(latent_dim, out_dim, latent_dim, mlp_layers, dropout, layer_norm=False)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.node_encoder(node_features)
        e = self.edge_encoder(edge_attr)
        for block in self.processor:
            h, e = block(h, e, edge_index)
        return self.decoder(h)

    def freeze_encoder_processor(self) -> None:
        for module in [self.node_encoder, self.edge_encoder, self.processor]:
            for p in module.parameters():
                p.requires_grad = False

    def freeze_encoder(self) -> None:
        for module in [self.node_encoder, self.edge_encoder]:
            for p in module.parameters():
                p.requires_grad = False

    def freeze_for_finetune(self, mode: str = "full") -> None:
        """Apply supported fine-tuning modes.

        mode:
        - full: train all parameters
        - decoder_only: freeze encoder + processor, train decoder
        - processor_decoder: freeze encoder, train processor + decoder
        """
        mode = (mode or "full").strip().lower()
        for p in self.parameters():
            p.requires_grad = True
        if mode in {"full", "all"}:
            return
        if mode in {"decoder_only", "freeze_encoder_processor"}:
            self.freeze_encoder_processor()
            return
        if mode in {"processor_decoder", "freeze_encoder"}:
            self.freeze_encoder()
            return
        raise ValueError("fine-tune mode must be full | decoder_only | processor_decoder")
