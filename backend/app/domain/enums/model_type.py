from __future__ import annotations

from enum import Enum


class ModelType(str, Enum):
    MESHGRAPHNET = "meshgraphnet"
    FNO = "fno"
    PINN = "pinn"

    @property
    def label(self) -> str:
        return {
            ModelType.MESHGRAPHNET: "MeshGraphNet",
            ModelType.FNO: "FNO",
            ModelType.PINN: "PINN",
        }[self]
