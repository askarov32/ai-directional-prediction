"""Graph construction from mesh elements or kNN fallback."""
from __future__ import annotations

import warnings
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def build_graph_from_mesh(coords: np.ndarray, elements: Dict[str, np.ndarray] | None = None, k_nearest: int = 12) -> Tuple[torch.Tensor, torch.Tensor]:
    coords = np.asarray(coords, dtype=np.float32)
    elements = elements or {}
    edges = set()

    for _, arr in elements.items():
        if arr is None or len(arr) == 0:
            continue
        for el in arr:
            nodes = [int(x) for x in el]
            for a in range(len(nodes)):
                for b in range(a + 1, len(nodes)):
                    i, j = nodes[a], nodes[b]
                    if 0 <= i < len(coords) and 0 <= j < len(coords) and i != j:
                        edges.add((min(i, j), max(i, j)))

    if not edges:
        warnings.warn(f"No usable mesh elements found. Building kNN graph with k={k_nearest}.")
        k = min(k_nearest + 1, len(coords))
        nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
        _, idx = nbrs.kneighbors(coords)
        for i, neighs in enumerate(idx):
            for j in neighs[1:]:
                if i != int(j):
                    edges.add((min(i, int(j)), max(i, int(j))))

    src, dst = [], []
    for i, j in sorted(edges):
        src.extend([i, j])
        dst.extend([j, i])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    pos = torch.tensor(coords, dtype=torch.float32)
    diff = pos[edge_index[1]] - pos[edge_index[0]]
    dist = torch.linalg.norm(diff, dim=1, keepdim=True)
    edge_attr = torch.cat([diff, dist], dim=1)
    return edge_index, edge_attr
