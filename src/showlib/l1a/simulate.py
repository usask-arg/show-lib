"""Simulate L1A granules from spectral radiance with the calibration database's model.

The detector follows the `model` section of src/showlib/schemas/cal_db.yaml: each
exposure has Poisson shot noise on the photo- and dark-current electrons,
saturation at full well, Gaussian read noise, and digitization at the
calibration's gain on top of the dark offset, clipped to the ADC range; the image
is the mean over the averaged exposures.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from showlib.cal_db import CalibrationDatabase, bin_width
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
        Increasing vacuum wavenumbers [cm-1] at which ``radiance`` is given,
        integrated with the trapezoid rule (showlib.cal_db.bin_width).
        CalibrationDatabase.model_wavenumber() gives a suitable grid.
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
        its brightest pixel reaches ``full_well_fraction`` of its saturation level:
        the lower of the full well and the ADC range above the pixel's dark
        offset, in photo- plus dark-current electrons.
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
        The granule, with the noise-free image (including the dark signal, before
        saturation and digitization) and simulation settings in its
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
    widths = bin_width(wavenumber)
    # One row's forward operator at a time, applied to every image: each operator
    # is (num_columns, len(wavenumber)), too large to hold for every row at once
    count_rates = np.empty((radiance.shape[0], cal_db.num_rows, cal_db.num_columns))
    for row in range(cal_db.num_rows):
        operator = cal_db.forward_operator(row, wavenumber, widths, exposure_time=1.0)
        count_rates[:, row] = radiance[:, row] @ operator.T
    rng = np.random.default_rng(seed)

    images, noiseless, peak_fraction = [], [], []
    for count_rate, image_geometry in zip(count_rates, geometry, strict=True):
        image_exposure = (
            exposure_time
            if exposure_time is not None
            else detector.auto_exposure(count_rate, full_well_fraction)
        )
        photo_counts = count_rate * image_exposure
        expected = detector.expected_counts(photo_counts, image_exposure)

        if noise:
            image, saturated = detector.expose(
                photo_counts, image_exposure, num_averaged_exposures, rng
            )
        else:
            image, saturated = detector.clip(photo_counts, image_exposure)

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
        if detector.can_saturate:
            peak_fraction.append(detector.peak_fraction(photo_counts, image_exposure))

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
    """Detector response for one row-by-column image, in counts and electrons."""

    def __init__(
        self,
        gain: float,
        read_noise: float,
        dark_offset: np.ndarray,
        dark_current: np.ndarray,
        full_well: float | None,
        adc_bits: int | None,
    ):
        self.gain = gain
        self.read_noise = read_noise
        self.dark_offset = dark_offset
        self.dark_current = dark_current
        self.full_well = np.inf if full_well is None else full_well
        self.max_count = np.inf if adc_bits is None else 2**adc_bits - 1
        # Photo- plus dark-current electrons at which each pixel saturates
        self.capacity = np.minimum(
            self.full_well, (self.max_count - dark_offset) * gain
        )
        self.can_saturate = bool(np.all(np.isfinite(self.capacity)))

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
        offset = cal_db.static_dark(0.0)
        return cls(
            float(ds["detector_gain"]),
            float(ds["detector_read_noise"]),
            offset,
            cal_db.static_dark(1.0) - offset,
            float(ds["detector_full_well"]) if "detector_full_well" in ds else None,
            int(ds["detector_adc_bits"]) if "detector_adc_bits" in ds else None,
        )

    def _electrons(self, photo_counts: np.ndarray, exposure_time: float) -> np.ndarray:
        """Expected photo- plus dark-current electrons."""
        return (photo_counts + exposure_time * self.dark_current) * self.gain

    def auto_exposure(self, count_rate: np.ndarray, fraction: float) -> float:
        """Exposure time [s] that puts the brightest pixel at ``fraction`` of its
        saturation level."""
        electron_rate = self._electrons(count_rate, 1.0)
        return fraction / np.max(electron_rate / self.capacity)

    def peak_fraction(self, photo_counts: np.ndarray, exposure_time: float) -> float:
        """Largest expected electrons as a fraction of the pixel's saturation level."""
        return float(
            np.max(self._electrons(photo_counts, exposure_time) / self.capacity)
        )

    def expected_counts(
        self, photo_counts: np.ndarray, exposure_time: float
    ) -> np.ndarray:
        """Noise-free signal [count], before saturation: offset plus electrons / g."""
        return (
            self.dark_offset + self._electrons(photo_counts, exposure_time) / self.gain
        )

    def _digitize(
        self, well: np.ndarray, read_noise: np.ndarray | float, rounding: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Counts for electrons collected in the well plus read noise, and where the
        well is full or the ADC is at its maximum."""
        counts = (
            self.dark_offset
            + (np.minimum(well, self.full_well) + read_noise) / self.gain
        )
        if rounding:
            counts = np.rint(counts)
        saturated = (well >= self.full_well) | (counts >= self.max_count)
        return np.clip(counts, 0, self.max_count), saturated

    def clip(
        self, photo_counts: np.ndarray, exposure_time: float
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Noise-free counts limited by full well and the ADC, and where they were."""
        if not self.can_saturate:
            return self.expected_counts(photo_counts, exposure_time), None
        electrons = self._electrons(photo_counts, exposure_time)
        return self._digitize(electrons, 0.0, rounding=False)

    def expose(
        self,
        photo_counts: np.ndarray,
        exposure_time: float,
        num_exposures: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean digitized counts over the exposures, and where any saturated."""
        expected_electrons = self._electrons(photo_counts, exposure_time)
        total = np.zeros_like(expected_electrons)
        saturated = np.zeros(expected_electrons.shape, dtype=bool)
        for _ in range(num_exposures):
            well = rng.poisson(expected_electrons).astype(float)
            read_noise = rng.normal(0.0, self.read_noise, well.shape)
            counts, exposure_saturated = self._digitize(well, read_noise, rounding=True)
            saturated |= exposure_saturated
            total += counts
        return total / num_exposures, saturated
