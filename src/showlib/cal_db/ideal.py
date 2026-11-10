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
    """Flat passband with raised-cosine transitions and a constant stopband.

    Parameters
    ----------
    wavelength_nm : array_like
        Wavelengths at which to evaluate the transmission [nm]
    centre_nm : float
        Centre of the passband [nm]
    half_width_nm : float
        Half-width of the flat passband [nm]; the passband is
        centre_nm ± half_width_nm
    transition_nm : float
        Width of the raised-cosine transition on each side of the passband [nm]
    stopband_db : float
        Stopband attenuation [dB]; the stopband transmission is
        10**(-stopband_db / 20)

    Returns
    -------
    np.ndarray
        Transmission, between the stopband value and 1, with the shape of
        wavelength_nm
    """
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

    Every row has the same centred, uniform OPD grid and Littrow wavenumber, and no
    fringe phase, visibility loss, or dispersion. Samples are on the natural grid,
    littrow - k / (num_columns · opd_spacing) for k = 1 ... num_columns / 2 - 1,
    kept where the filter transmits at least ``min_sample_transmission``.

    Parameters
    ----------
    num_rows : int
        Number of detector rows (lines of sight)
    num_columns : int, optional
        Number of detector columns (OPD samples), by default 1280
    opd_spacing_cm : float, optional
        OPD step between adjacent columns [cm], by default 0.0075
    littrow_wavelength_nm : float, optional
        Littrow (heterodyne) wavelength [nm], by default 1362
    filter_centre_nm : float, optional
        Centre of the bandpass filter [nm], by default 1364
    filter_half_width_nm : float, optional
        Half-width of the filter's flat passband [nm], by default 1.5
    filter_transition_nm : float, optional
        Width of each of the filter's raised-cosine edges [nm], by default 0.2
    filter_stopband_db : float, optional
        Filter stopband attenuation [dB], by default 80
    min_sample_transmission : float, optional
        Smallest filter transmission at which a natural-grid wavenumber becomes an
        L1B sample, by default 0.01
    table_margin_nm : float, optional
        How far the spectral_transmission table extends beyond the filter edges
        [nm], by default 0.5
    table_resolution_cminv : float, optional
        Spacing of the spectral_transmission table [cm-1], by default 0.02
    pixel_etendue : float, optional
        Effective etendue of every pixel [m2 sr], by default 1e-11
    detector_full_well : float, optional
        Full-well capacity [electrons], by default 450000
    detector_adc_bits : int, optional
        ADC resolution [bits], by default 12. The detector gain is
        detector_full_well / 2**detector_adc_bits electrons per count.
    detector_read_noise : float, optional
        RMS read noise per exposure [electrons], by default 75
    valid_from : str, optional
        Start of the calibration's validity period, as an ISO 8601 UTC timestamp,
        by default "2026-01-01T00:00:00Z"

    Returns
    -------
    CalibrationDatabase
        The validated calibration database
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
