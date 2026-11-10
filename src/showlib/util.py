from __future__ import annotations

import numpy as np


def wrap_degrees(angle, lower: float) -> np.ndarray:
    """Wrap angles in degrees into [lower, lower + 360)."""
    return (np.asarray(angle, dtype=float) - lower) % 360.0 + lower
