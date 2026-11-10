"""An ideal SHOW calibration database: centred uniform OPD and a bandpass filter.

Defaults follow the hawc-simulator SHOW configuration. Useful for simulations and
tests until a measured calibration exists.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from showlib.cal_db import CalibrationDatabase
from showlib.schemas import apply_metadata


def bandpass_filter(
    wavelength_nm,
    centre_nm: float,
    half_width_nm: float,
    transition_nm: float,
    stopband_db: float,
) -> np.ndarray:
    """Flat passband with raised-cosine transitions and a constant stopband."""
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    stopband = 10 ** (-stopband_db / 20)
    distance = np.abs(wavelength_nm - centre_nm) - half_width_nm

    transmission = np.full(wavelength_nm.shape, stopband)
    transmission[distance <= 0] = 1.0
    transition = (distance > 0) & (distance <= transition_nm)
    transmission[transition] = stopband + 0.5 * (1 - stopband) * (
        1 + np.cos(np.pi * distance[transition] / transition_nm)
    )
    return transmission


def ideal_calibration_database(
    num_rows: int,
    num_columns: int = 1280,
    opd_spacing_cm: float = 0.0075,
    littrow_wavelength_nm: float = 1362.0,
    filter_centre_nm: float = 1364.0,
    filter_half_width_nm: float = 1.5,
    filter_transition_nm: float = 0.2,
    filter_stopband_db: float = 80.0,
    min_sample_transmission: float = 0.01,
    table_margin_nm: float = 0.5,
    table_resolution_cminv: float = 0.02,
    pixel_etendue: float = 1e-11,
    detector_full_well: float = 450_000.0,
    detector_adc_bits: int = 12,
    detector_read_noise: float = 75.0,
    valid_from: str = "2026-01-01T00:00:00Z",
) -> CalibrationDatabase:
    """Build an ideal calibration database.

    Samples are on the natural grid, littrow - k / (num_columns · opd_spacing) for
    k = 1 ... num_columns / 2 - 1, kept where the filter transmits at least
    ``min_sample_transmission``. The detector gain is the full well divided by the
    ADC range.
    """
    littrow = 1e7 / littrow_wavelength_nm
    natural_spacing = 1 / (num_columns * opd_spacing_cm)

    filter_extent = filter_half_width_nm + filter_transition_nm + table_margin_nm
    table = np.arange(
        1e7 / (filter_centre_nm + filter_extent),
        1e7 / (filter_centre_nm - filter_extent) + table_resolution_cminv,
        table_resolution_cminv,
    )

    def transmission(wavenumber):
        return bandpass_filter(
            1e7 / wavenumber,
            filter_centre_nm,
            filter_half_width_nm,
            filter_transition_nm,
            filter_stopband_db,
        )

    samples = littrow - natural_spacing * np.arange(1, num_columns // 2)
    keep = (
        (samples >= table[0])
        & (samples <= table[-1])
        & (transmission(samples) >= min_sample_transmission)
    )
    samples = np.sort(samples[keep])

    opd_edges = (np.arange(num_columns + 1) - num_columns / 2) * opd_spacing_cm
    ds = xr.Dataset(
        {
            "opd": (
                ["los_edge", "pixelcolumn_edge"],
                np.broadcast_to(opd_edges, (num_rows + 1, num_columns + 1)).copy(),
            ),
            "littrow_wavenumber": (["los_edge"], np.full(num_rows + 1, littrow)),
            "pixel_etendue": (
                ["los", "pixelcolumn"],
                np.full((num_rows, num_columns), pixel_etendue),
            ),
            "spectral_transmission": (["transmission_wavenumber"], transmission(table)),
            "detector_gain": ((), detector_full_well / 2**detector_adc_bits),
            "detector_read_noise": ((), float(detector_read_noise)),
            "detector_full_well": ((), float(detector_full_well)),
            "detector_adc_bits": ((), np.int64(detector_adc_bits)),
            "sample_wavenumber": (
                ["los", "spectral_sample"],
                np.broadcast_to(samples, (num_rows, len(samples))).copy(),
            ),
        },
        coords={"transmission_wavenumber": table},
        attrs={"valid_from": valid_from},
    )
    return CalibrationDatabase(apply_metadata(ds, "cal_db"))
