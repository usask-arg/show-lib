"""Simulate L1B granules from spectral radiance."""

from __future__ import annotations

import numpy as np

from showlib.cal_db import CalibrationDatabase
from showlib.l1a.simulate import simulate_l1a
from showlib.l1b.data import L1bDataSet
from showlib.processing.l1a_to_l1b import process_l1a_to_l1b


def simulate_l1b(
    cal_db: CalibrationDatabase,
    wavenumber: np.ndarray,
    radiance: np.ndarray,
    geometry: list[dict],
    *,
    apodization: float | None = 10.0,
    noise_fraction: float = 0.0,
    **l1a_kwargs,
) -> L1bDataSet:
    """Simulate an L1B granule: simulate_l1a followed by process_l1a_to_l1b.

    The L1B therefore has exactly the radiance, line shapes, noise and noise
    correlation that the processing chain gives. ``wavenumber``, ``radiance`` and
    ``geometry`` are as for simulate_l1a, and ``l1a_kwargs`` (exposure_time,
    full_well_fraction, num_averaged_exposures, noise, seed) are passed to it;
    ``apodization`` and ``noise_fraction`` to process_l1a_to_l1b.
    """
    l1a = simulate_l1a(cal_db, wavenumber, radiance, geometry, **l1a_kwargs)
    return process_l1a_to_l1b(
        l1a, cal_db, apodization=apodization, noise_fraction=noise_fraction
    )
