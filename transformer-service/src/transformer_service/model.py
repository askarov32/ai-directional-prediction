from __future__ import annotations

import torch
from torch import Tensor, nn


ACTIVATION_MAP: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


def _resolve_activation(name: str) -> type[nn.Module]:
    if name not in ACTIVATION_MAP:
        raise ValueError(f"Unsupported activation: {name}")
    return ACTIVATION_MAP[name]


def _build_ffn(d_model: int, expansion: int, dropout: float, activation: str) -> nn.Sequential:
    activation_layer = _resolve_activation(activation)
    return nn.Sequential(
        nn.Linear(d_model, d_model * expansion),
        activation_layer(),
        nn.Dropout(dropout),
        nn.Linear(d_model * expansion, d_model),
        nn.Dropout(dropout),
    )


class OFormerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_expansion: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = _build_ffn(d_model, ffn_expansion, dropout, activation)

    def forward(self, tokens: Tensor) -> Tensor:
        attn_in = self.norm_attn(tokens)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        tokens = tokens + attn_out
        ffn_out = self.ffn(self.norm_ffn(tokens))
        tokens = tokens + ffn_out
        return tokens


class OFormerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_expansion: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = _build_ffn(d_model, ffn_expansion, dropout, activation)

    def forward(self, queries: Tensor, encoded: Tensor) -> Tensor:
        q_n = self.norm_q(queries)
        kv_n = self.norm_kv(encoded)
        attn_out, _ = self.cross_attn(q_n, kv_n, kv_n, need_weights=False)
        queries = queries + attn_out
        ffn_out = self.ffn(self.norm_ffn(queries))
        queries = queries + ffn_out
        return queries


class OFormer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        query_dim: int,
        output_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        enc_depth: int = 4,
        dec_depth: int = 4,
        ffn_expansion: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.query_proj = nn.Linear(query_dim, d_model)
        self.encoder = nn.ModuleList(
            [
                OFormerEncoderLayer(d_model, n_heads, ffn_expansion, dropout, activation)
                for _ in range(enc_depth)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                OFormerDecoderLayer(d_model, n_heads, ffn_expansion, dropout, activation)
                for _ in range(dec_depth)
            ]
        )
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, input_tokens: Tensor, query_coords: Tensor) -> Tensor:
        if input_tokens.dim() == 2:
            input_tokens = input_tokens.unsqueeze(0)
        if query_coords.dim() == 2:
            query_coords = query_coords.unsqueeze(0)
        encoded = self.input_proj(input_tokens)
        for layer in self.encoder:
            encoded = layer(encoded)
        queries = self.query_proj(query_coords)
        for layer in self.decoder:
            queries = layer(queries, encoded)
        queries = self.head_norm(queries)
        return self.head(queries)
