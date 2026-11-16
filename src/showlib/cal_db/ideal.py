"""An ideal SHOW calibration database: centred uniform OPD and a bandpass filter.

Defaults follow the hawc-simulator SHOW configuration. The dark signal defaults are
placeholders, not measurements. Useful for simulations and tests until a measured
calibration exists.
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
    dark_offset: float = 100.0,
    dark_offset_spread: float = 5.0,
    dark_current: float = 10.0,
    dark_seed: int = 0,
    fringe_visibility: float = 1.0,
    grating_tilt_rad: float = 0.0,
    tilted_grating: int = 1,
    row_pitch_cm: float | None = None,
    magnification: float = 1.0,
    row_edge_height_cm: np.ndarray | None = None,
    valid_from: str = "2026-01-01T00:00:00Z",
) -> CalibrationDatabase:
    """Build an ideal calibration database.

    Every row has the same Littrow wavenumber and no dispersion. Without a grating
    tilt, every row has the same centred, uniform OPD grid and no fringe phase.
    Samples are on the natural grid, littrow - k / (num_columns · opd_spacing) for
    k = 1 ... num_columns / 2 - 1, kept where the filter transmits at least
    ``min_sample_transmission``.

    A tilt ε of one grating about the dispersion axis adds ±2εy to the OPD and
    ±2π · littrow · 2εy to the fringe phase at each pixel corner, where y is the
    height of the corner's row edge on the grating (see the grating-tilt part of
    the `model` section of cal_db.yaml). The heights come from
    ``row_edge_height_cm``, or else from ``row_pitch_cm`` and ``magnification``
    with the rows centred on y = 0.

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
    dark_offset : float, optional
        Mean dark offset (bias) [count], by default 100 (a placeholder)
    dark_offset_spread : float, optional
        Standard deviation of the dark offset from pixel to pixel [count], by
        default 5 (a placeholder). Offsets are clipped at 0.
    dark_current : float, optional
        Dark current of every pixel [count s-1], by default 10 (a placeholder);
        0 omits dark_current from the database
    dark_seed : int, optional
        Seed for the pixel-to-pixel dark offset pattern, by default 0
    fringe_visibility : float, optional
        Intrinsic fringe contrast of every pixel, by default 1 (omitted from the
        database)
    grating_tilt_rad : float, optional
        Tilt ε of one grating about the dispersion axis [rad], by default 0
    tilted_grating : int, optional
        Which grating is tilted: 1 for the grating in the arm whose path counts
        positive in opd (adds +2εy to the OPD), 2 for the other (adds -2εy), by
        default 1
    row_pitch_cm : float, optional
        Height of one detector row (line of sight) on the detector [cm]. Needed
        for a grating tilt unless row_edge_height_cm is given.
    magnification : float, optional
        Detector-to-grating magnification: a height y on the grating is imaged at
        magnification · y on the detector, by default 1
    row_edge_height_cm : array_like, optional
        Height y of each row edge on the grating [cm], length num_rows + 1.
        Overrides row_pitch_cm and magnification.
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
    opd = np.broadcast_to(opd_edges, (num_rows + 1, num_columns + 1)).copy()
    pixel_shape = (num_rows, num_columns)
    rng = np.random.default_rng(dark_seed)
    data_vars = {
        "littrow_wavenumber": (["los_edge"], np.full(num_rows + 1, littrow)),
        "pixel_etendue": (["los", "pixelcolumn"], np.full(pixel_shape, pixel_etendue)),
        "spectral_transmission": (["transmission_wavenumber"], transmission(table)),
        "dark_offset": (
            ["los", "pixelcolumn"],
            np.clip(
                dark_offset + dark_offset_spread * rng.standard_normal(pixel_shape),
                0,
                None,
            ),
        ),
    }
    if dark_current:
        data_vars["dark_current"] = (
            ["los", "pixelcolumn"],
            np.full(pixel_shape, float(dark_current)),
        )
    if fringe_visibility != 1:
        data_vars["fringe_visibility"] = (
            ["los", "pixelcolumn"],
            np.full(pixel_shape, float(fringe_visibility)),
        )
    if grating_tilt_rad:
        tilt_opd = _tilt_opd(
            num_rows,
            grating_tilt_rad,
            tilted_grating,
            row_pitch_cm,
            magnification,
            row_edge_height_cm,
        )
        opd += tilt_opd[:, np.newaxis]
        data_vars["fringe_phase"] = (
            ["los_edge", "pixelcolumn_edge"],
            np.broadcast_to(
                2 * np.pi * littrow * tilt_opd[:, np.newaxis], opd.shape
            ).copy(),
        )

    ds = xr.Dataset(
        {
            "opd": (["los_edge", "pixelcolumn_edge"], opd),
            **data_vars,
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


def _tilt_opd(
    num_rows: int,
    tilt_rad: float,
    tilted_grating: int,
    row_pitch_cm: float | None,
    magnification: float,
    row_edge_height_cm: np.ndarray | None,
) -> np.ndarray:
    """OPD added at each row edge by a grating tilt, ±2εy [cm]."""
    if tilted_grating not in (1, 2):
        msg = f"tilted_grating must be 1 or 2, got {tilted_grating}"
        raise ValueError(msg)
    if row_edge_height_cm is not None:
        height = np.asarray(row_edge_height_cm, dtype=float)
        if height.shape != (num_rows + 1,):
            msg = (
                f"row_edge_height_cm must have length num_rows + 1 = {num_rows + 1}, "
                f"got shape {height.shape}"
            )
            raise ValueError(msg)
    elif row_pitch_cm is not None:
        if not magnification > 0:
            msg = f"magnification must be positive, got {magnification}"
            raise ValueError(msg)
        height = (np.arange(num_rows + 1) - num_rows / 2) * row_pitch_cm / magnification
    else:
        msg = "A grating tilt needs row_pitch_cm or row_edge_height_cm"
        raise ValueError(msg)
    sign = 1.0 if tilted_grating == 1 else -1.0
    return sign * 2 * tilt_rad * height
