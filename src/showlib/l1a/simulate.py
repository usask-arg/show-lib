"""Simulate L1A granules from spectral radiance with the calibration database's model.

The detector follows the hawc-simulator SHOW model: each exposure has Poisson shot
noise, saturation at full well, Gaussian read noise, and digitization at the
calibration's gain, clipped to the ADC range; the image is the mean over the
averaged exposures.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from showlib.cal_db import CalibrationDatabase
from showlib.l1a.data import L1ADataSet, L1AImage


def simulate_l1a(
    cal_db: CalibrationDatabase,
    wavenumber: np.ndarray,
    radiance: np.ndarray,
    geometry: list[dict],
    *,
    exposure_time: float | None = None,
    full_well_fraction: float = 0.8,
    num_averaged_exposures: int = 1,
    noise: bool = True,
    seed: int | None = None,
) -> L1ADataSet:
    """Simulate an L1A granule.

    Parameters
    ----------
    cal_db : CalibrationDatabase
        Instrument model and detector parameters
    wavenumber : np.ndarray
        Increasing vacuum wavenumbers [cm-1] at which ``radiance`` is given. Each
        value represents a bin of width np.gradient(wavenumber).
    radiance : np.ndarray
        Spectral radiance [W m-2 sr-1 nm-1], shape
        (num_images, num_rows, len(wavenumber)), one spectrum per detector row
    geometry : list[dict]
        For each image, the geolocation keyword arguments of L1AImage: time,
        tangent_altitude, tangent_latitude, tangent_longitude, spacecraft_latitude,
        spacecraft_longitude, spacecraft_altitude, solar_zenith_angle,
        relative_solar_azimuth_angle, los_azimuth_angle
    exposure_time : float, optional
        Single-exposure integration time [s]. By default each image is exposed so
        its brightest pixel reaches ``full_well_fraction`` of full well.
    full_well_fraction : float, optional
        Auto-exposure target for the brightest pixel, by default 0.8
    num_averaged_exposures : int, optional
        Number of equal exposures averaged into each image, by default 1
    noise : bool, optional
        Apply shot, read, and digitization noise, by default True
    seed : int, optional
        Seed for the noise realization

    Returns
    -------
    L1ADataSet
        The granule, with the noise-free image and simulation settings in its
        ``simulation`` group
    """
    radiance = np.asarray(radiance, dtype=float)
    wavenumber = np.asarray(wavenumber, dtype=float)
    expected_shape = (cal_db.num_rows, len(wavenumber))
    if radiance.ndim != 3 or radiance.shape[1:] != expected_shape:
        msg = (
            f"radiance must have shape (num_images, {expected_shape[0]}, "
            f"{expected_shape[1]}), got {radiance.shape}"
        )
        raise ValueError(msg)
    if len(geometry) != radiance.shape[0]:
        msg = f"Expected geometry for {radiance.shape[0]} images, got {len(geometry)}"
        raise ValueError(msg)

    detector = _Detector.from_calibration(cal_db, noise or exposure_time is None)
    bin_width = np.gradient(wavenumber)
    count_rate_operators = [
        cal_db.forward_operator(row, wavenumber, bin_width, exposure_time=1.0)
        for row in range(cal_db.num_rows)
    ]
    rng = np.random.default_rng(seed)

    images, noiseless, peak_fraction = [], [], []
    for image_radiance, image_geometry in zip(radiance, geometry, strict=True):
        count_rate = np.stack(
            [
                operator @ spectrum
                for operator, spectrum in zip(
                    count_rate_operators, image_radiance, strict=True
                )
            ]
        )
        image_exposure = (
            exposure_time
            if exposure_time is not None
            else full_well_fraction * detector.full_well_counts / count_rate.max()
        )
        expected = count_rate * image_exposure

        saturated = None
        image = expected
        if noise:
            image, saturated = detector.expose(expected, num_averaged_exposures, rng)

        images.append(
            L1AImage(
                image=image,
                exposure_time=image_exposure,
                num_averaged_exposures=num_averaged_exposures,
                saturated=saturated,
                **image_geometry,
            )
        )
        noiseless.append(expected)
        if detector.full_well_counts is not None:
            peak_fraction.append(expected.max() / detector.full_well_counts)

    simulation = xr.Dataset(
        {
            "image_noiseless": (["time", "los", "pixelcolumn"], np.stack(noiseless)),
            "noise_enabled": ((), noise),
        }
    )
    if peak_fraction:
        simulation["expected_peak_full_well_fraction"] = ("time", peak_fraction)
    if exposure_time is None:
        simulation["auto_exposure_full_well_fraction"] = (
            "time",
            np.full(len(images), full_well_fraction),
        )
    if noise and seed is not None:
        simulation["noise_seed"] = ((), np.int64(seed))

    return L1ADataSet.from_images(images, simulation=simulation)


class _Detector:
    def __init__(
        self,
        gain: float,
        read_noise: float,
        full_well: float | None,
        adc_bits: int | None,
    ):
        self.gain = gain
        self.read_noise = read_noise
        self.full_well = full_well
        self.max_count = None if adc_bits is None else 2**adc_bits - 1
        self.full_well_counts = None if full_well is None else full_well / gain

    @classmethod
    def from_calibration(cls, cal_db: CalibrationDatabase, needs_limits: bool):
        ds = cal_db.ds
        if needs_limits:
            missing = [
                name
                for name in ("detector_full_well", "detector_adc_bits")
                if name not in ds
            ]
            if missing:
                msg = (
                    "Noise and auto-exposure need the calibration database's "
                    f"{' and '.join(missing)}"
                )
                raise ValueError(msg)
        return cls(
            float(ds["detector_gain"]),
            float(ds["detector_read_noise"]),
            float(ds["detector_full_well"]) if "detector_full_well" in ds else None,
            int(ds["detector_adc_bits"]) if "detector_adc_bits" in ds else None,
        )

    def expose(
        self, expected_counts: np.ndarray, num_exposures: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean digitized counts over the exposures, and where any saturated."""
        expected_electrons = expected_counts * self.gain
        total = np.zeros_like(expected_counts)
        saturated = np.zeros(expected_counts.shape, dtype=bool)
        for _ in range(num_exposures):
            electrons = rng.poisson(expected_electrons).astype(float)
            saturated |= electrons >= self.full_well
            electrons = np.minimum(electrons, self.full_well) + rng.normal(
                0.0, self.read_noise, electrons.shape
            )
            saturated |= electrons >= self.full_well
            electrons = np.clip(electrons, 0.0, self.full_well)
            counts = np.clip(np.rint(electrons / self.gain), 0, self.max_count)
            total += counts
        return total / num_exposures, saturated
