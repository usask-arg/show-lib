"""L1A to L1B: invert SHOW interferograms into calibrated radiance spectra.

The forward model A and the inverse B are described in the `model` section of
src/showlib/schemas/cal_db.yaml.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from showlib.cal_db import CalibrationDatabase
from showlib.l1a.data import L1ADataSet
from showlib.l1b.data import L1bDataSet
from showlib.schemas import apply_metadata, check

_GEOMETRY = (
    "tangent_altitude",
    "tangent_latitude",
    "tangent_longitude",
    "spacecraft_latitude",
    "spacecraft_longitude",
    "spacecraft_altitude",
    "solar_zenith_angle",
    "relative_solar_azimuth_angle",
    "los_azimuth_angle",
)


def inverse_operator(
    cal_db: CalibrationDatabase, row: int, apodization: float | None = 10.0
) -> np.ndarray:
    """B for one detector row, shape (num_samples, num_columns).

    Maps the image count rate [counts s-1] of the row to radiance
    [W m-2 sr-1 nm-1] at the row's sample wavenumbers. The count rate is
    flat-fielded and converted to photon rate per unit etendue, its mean is
    removed, the apodization window is applied over OPD, and the pixel-averaged
    fringe model (with a DC term) is fit by least squares. With apodization=None
    the inverse is exact for spectra that lie on the sample grid.
    """
    samples = cal_db.sample_wavenumber(row)
    if len(samples) < 2:
        msg = "Each detector row needs at least two spectral samples"
        raise ValueError(msg)

    model = np.column_stack([np.ones(cal_db.num_columns), cal_db.fringe(row, samples)])
    fit = np.linalg.pinv(model)[1:] * _window(cal_db.pixel_centre_opd(row), apodization)
    fit -= fit.mean(axis=1, keepdims=True)

    gain = float(cal_db.ds["detector_gain"])
    etendue = cal_db.ds["pixel_etendue"].to_numpy()[row]
    weight = cal_db.radiance_weight(samples, np.gradient(samples))
    return fit * (gain / etendue)[np.newaxis, :] / weight[:, np.newaxis]


def process_l1a_to_l1b(
    l1a: L1ADataSet,
    cal_db: CalibrationDatabase,
    apodization: float | None = 10.0,
    noise_fraction: float = 0.0,
) -> L1bDataSet:
    """Process an L1A granule into an L1B granule.

    apodization is the Kaiser window beta (None for a rectangular window).
    radiance_noise propagates shot, read, and quantization noise through B, and
    adds noise_fraction * |radiance| in quadrature.
    """
    if apodization is not None and (not np.isfinite(apodization) or apodization < 0):
        msg = "apodization must be None or a finite, non-negative Kaiser beta"
        raise ValueError(msg)
    if not np.isfinite(noise_fraction) or noise_fraction < 0:
        msg = "noise_fraction must be finite and non-negative"
        raise ValueError(msg)

    ds = l1a.ds
    check(ds, "l1a", cal_db=cal_db.ds)

    image = ds["image"].transpose("time", "los", "pixelcolumn").to_numpy()
    exposure_time = ds["exposure_time"].to_numpy()[:, np.newaxis, np.newaxis]
    num_exposures = (
        ds["num_averaged_exposures"].to_numpy()[:, np.newaxis, np.newaxis]
        if "num_averaged_exposures" in ds
        else 1
    )
    gain = float(cal_db.ds["detector_gain"])
    read_noise = float(cal_db.ds["detector_read_noise"])

    count_rate = image / exposure_time
    count_variance = (
        np.clip(image * gain, 0, None) + read_noise**2 + gain**2 / 12
    ) / (num_exposures * gain**2)
    rate_variance = count_variance / exposure_time**2

    num_samples = cal_db.ds.sizes["spectral_sample"]
    shape = (ds.sizes["time"], ds.sizes["los"], num_samples)
    radiance = np.empty(shape)
    detector_noise = np.empty(shape)
    for row in range(ds.sizes["los"]):
        operator = inverse_operator(cal_db, row, apodization)
        radiance[:, row] = count_rate[:, row] @ operator.T
        detector_noise[:, row] = np.sqrt(rate_variance[:, row] @ (operator**2).T)
    radiance_noise = np.hypot(detector_noise, noise_fraction * np.abs(radiance))

    if "saturated" in ds:
        saturated_fraction = ds["saturated"].mean("pixelcolumn").to_numpy()
    else:
        saturated_fraction = np.zeros((ds.sizes["time"], ds.sizes["los"]))

    attrs = {
        "apodization_window": "rectangular" if apodization is None else "kaiser",
        "radiance_noise_model": "detector+fractional" if noise_fraction else "detector",
    }
    if apodization is not None:
        attrs["apodization_beta"] = float(apodization)

    spectral_dims = ["time", "los", "spectral_sample"]
    l1b = xr.Dataset(
        {
            "radiance": (spectral_dims, radiance),
            "radiance_noise": (spectral_dims, radiance_noise),
            "saturated_fraction": (["time", "los"], saturated_fraction),
            **{name: ds[name] for name in _GEOMETRY},
        },
        coords={
            "time": ds["time"],
            "los": np.arange(ds.sizes["los"]),
            "wavenumber": cal_db.ds["sample_wavenumber"].variable,
        },
        attrs=attrs,
    )
    return L1bDataSet(apply_metadata(l1b, "l1b"))


def _window(opd: np.ndarray, apodization: float | None) -> np.ndarray:
    """Kaiser window evaluated over OPD; equals np.kaiser on a uniform grid."""
    if apodization is None:
        return np.ones_like(opd)
    centre = 0.5 * (opd.max() + opd.min())
    half_width = 0.5 * (opd.max() - opd.min())
    position = (opd - centre) / half_width
    return np.i0(apodization * np.sqrt(np.clip(1 - position**2, 0, None))) / np.i0(
        apodization
    )
