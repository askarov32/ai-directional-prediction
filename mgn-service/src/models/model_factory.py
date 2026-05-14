"""Model factory wrapper."""
from __future__ import annotations

from typing import Dict

from .meshgraphnet import ConditionalMeshGraphNet


def create_model(config: Dict, node_in_dim: int, edge_in_dim: int, out_dim: int) -> ConditionalMeshGraphNet:
    mcfg = config.get("model", {})
    model_type = str(mcfg.get("type", "conditional_meshgraphnet")).lower()
    if model_type not in {"conditional_meshgraphnet", "meshgraphnet"}:
        raise ValueError(f"Unsupported model type: {model_type}")
    return ConditionalMeshGraphNet(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        out_dim=out_dim,
        latent_dim=int(mcfg.get("latent_dim", 128)),
        message_passing_steps=int(mcfg.get("message_passing_steps", 10)),
        mlp_layers=int(mcfg.get("mlp_layers", 3)),
        dropout=float(mcfg.get("dropout", 0.05)),
        layer_norm=bool(mcfg.get("layer_norm", True)),
    )
