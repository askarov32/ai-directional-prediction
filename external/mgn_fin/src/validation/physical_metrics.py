"""Physical metric helpers used by validation/reporting."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .validation import compute_derived_fields


def derived_fields(trajectory: np.ndarray, field_names: List[str], coords: np.ndarray | None = None) -> Dict[str, np.ndarray]:
    return compute_derived_fields(trajectory, field_names, coords)
