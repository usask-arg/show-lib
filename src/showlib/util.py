from __future__ import annotations

import numpy as np


def wrap_degrees(angle, lower: float) -> np.ndarray:
    """Wrap angles in degrees into [lower, lower + 360)."""
    wrapped = (np.asarray(angle, dtype=float) - lower) % 360.0 + lower
    # An angle just below lower wraps to lower + 360 after rounding
    return np.where(wrapped >= lower + 360.0, lower, wrapped)


def circular_mean_degrees(angle) -> float:
    """Mean of angles in degrees taken on the circle, in [-180, 180).

    A plain mean of longitudes either side of 180 (or of 0, in [0, 360)) lands on
    the far side of the Earth.
    """
    radians = np.deg2rad(np.asarray(angle, dtype=float))
    mean = np.rad2deg(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean()))
    return float(wrap_degrees(mean, -180.0))
