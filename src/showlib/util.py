from __future__ import annotations

import numpy as np


def wrap_degrees(angle, lower: float) -> np.ndarray:
    """Wrap angles in degrees into [lower, lower + 360)."""
    wrapped = (np.asarray(angle, dtype=float) - lower) % 360.0 + lower
    # An angle just below lower wraps to lower + 360 after rounding
    return np.where(wrapped >= lower + 360.0, lower, wrapped)
