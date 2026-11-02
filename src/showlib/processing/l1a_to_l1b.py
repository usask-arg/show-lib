from __future__ import annotations

import numpy as np
import xarray as xr

from showlib.cal_db import CalibrationDatabase
from showlib.l1a.data import L1AImage
from showlib.l1b.data import L1bImage

_PLANCK_CONSTANT_J_S = 6.62607015e-34
_SPEED_OF_LIGHT_M_S = 299792458.0


def _scalar(ds: xr.Dataset, name: str, default: float | None = None) -> float:
    if name not in ds:
        if default is not None:
            return default
        msg = f"L1A data must contain {name}"
        raise ValueError(msg)

    values = np.asarray(ds[name].to_numpy()).reshape(-1)
    if len(values) != 1 or not np.isfinite(values[0]):
        msg = f"L1A field {name} must be a finite scalar"
        raise ValueError(msg)
    return float(values[0])


def _filter_transmission(
    cal_db: xr.Dataset, sample_wavenumber: np.ndarray
) -> np.ndarray:
    if "hires_wavenumber" not in cal_db or "filter" not in cal_db:
        msg = "Calibration database must contain hires_wavenumber and filter"
        raise ValueError(msg)

    filter_wavenumber = cal_db["hires_wavenumber"].to_numpy()
    transmission = cal_db["filter"].to_numpy()
    order = np.argsort(filter_wavenumber)
    result = np.interp(
        sample_wavenumber,
        filter_wavenumber[order],
        transmission[order],
        left=0.0,
        right=0.0,
    )
    if np.any(result <= 0):
        msg = "Calibration filter transmission must be positive at every sample"
        raise ValueError(msg)
    return result


def _spectral_gain(ds: xr.Dataset, wavelength_nm: np.ndarray) -> np.ndarray:
    scalar_gain = _scalar(ds, "interferogram_gain", 1.0)
    if "column_etendue_m2_sr" not in ds:
        return np.full_like(wavelength_nm, scalar_gain, dtype=float)

    photon_energy_j = (
        _PLANCK_CONSTANT_J_S * _SPEED_OF_LIGHT_M_S / (wavelength_nm * 1e-9)
    )
    exposure_time_s = _scalar(ds, "exposure_time_s")
    return (
        scalar_gain
        * _scalar(ds, "column_etendue_m2_sr")
        * exposure_time_s
        * _scalar(ds, "interferometer_port_fraction")
        * _scalar(ds, "system_efficiency")
        / photon_energy_j
    )


def _inverse_cosine_transform(
    image: np.ndarray,
    opd_cm: np.ndarray,
    relative_wavenumber: np.ndarray,
    apodization: float | None,
) -> np.ndarray:
    operator = _inverse_cosine_operator(
        image.shape[1], opd_cm, relative_wavenumber, apodization
    )
    return np.einsum("yn,sn->ys", image, operator)


def _inverse_cosine_operator(
    num_columns: int,
    opd_cm: np.ndarray,
    relative_wavenumber: np.ndarray,
    apodization: float | None,
) -> np.ndarray:
    basis = np.cos(
        2 * np.pi * relative_wavenumber[:, np.newaxis] * opd_cm[np.newaxis, :]
    )
    basis -= np.mean(basis, axis=1, keepdims=True)

    if apodization is None:
        window = np.ones(num_columns)
    else:
        window = np.kaiser(num_columns, apodization)

    basis_norm = np.einsum("sn,sn->s", basis, basis)
    if np.any(basis_norm <= np.finfo(float).eps):
        msg = "Calibration contains a spectral sample that cannot be separated from DC"
        raise ValueError(msg)

    weighted_basis = window[np.newaxis, :] * basis
    weighted_basis -= np.mean(weighted_basis, axis=1, keepdims=True)
    return weighted_basis / basis_norm[:, np.newaxis]


def _detector_pixel_variance(ds: xr.Dataset, image: np.ndarray) -> np.ndarray | None:
    if "detector_noise_estimate_available" in ds:
        estimate_available = bool(_scalar(ds, "detector_noise_estimate_available"))
    else:
        estimate_available = bool(_scalar(ds, "noise_enabled", 0.0))
    if not estimate_available:
        return None

    expected_electrons = np.asarray(
        ds["image_noiseless"].to_numpy() if "image_noiseless" in ds else image,
        dtype=float,
    )
    if expected_electrons.shape != image.shape:
        msg = "image_noiseless must have the same shape as the L1A image"
        raise ValueError(msg)
    if np.any(~np.isfinite(expected_electrons)) or np.any(expected_electrons < 0):
        msg = "Expected detector electrons must be finite and non-negative"
        raise ValueError(msg)

    read_noise_electrons = _scalar(ds, "detector_read_noise_electrons")
    electrons_per_dn = _scalar(ds, "detector_electrons_per_dn")
    if read_noise_electrons < 0:
        msg = "detector_read_noise_electrons must be non-negative"
        raise ValueError(msg)
    if electrons_per_dn <= 0:
        msg = "detector_electrons_per_dn must be greater than zero"
        raise ValueError(msg)

    num_averaged_exposures = _scalar(ds, "num_averaged_exposures", 1.0)
    if num_averaged_exposures < 1 or not float(num_averaged_exposures).is_integer():
        msg = "num_averaged_exposures must be a positive integer"
        raise ValueError(msg)

    return (
        expected_electrons + read_noise_electrons**2 + electrons_per_dn**2 / 12
    ) / num_averaged_exposures


def _saturated_fraction(ds: xr.Dataset, image: np.ndarray) -> np.ndarray | None:
    if "saturated" not in ds:
        return None

    saturated = np.asarray(ds["saturated"].to_numpy(), dtype=bool)
    if saturated.shape != image.shape:
        msg = "saturated must have the same shape as the L1A image"
        raise ValueError(msg)
    return np.mean(saturated, axis=1)


def process_l1a_to_l1b_dataset(
    l1a: L1AImage,
    cal_db: CalibrationDatabase,
    noise_fraction: float = 0.0,
    apodization: float | None = 10.0,
) -> L1bImage:
    """Invert a centered simple-SHOW interferogram into calibrated radiance.

    The inverse projects each detector row onto the centered cosine basis from
    the calibration spectral grid. By default, a Kaiser window with beta 10 is
    applied before the projection; pass ``apodization=None`` for a rectangular
    window. The unapodized transform normalization is retained so apodization
    broadens spectral lines without applying a coherent-gain correction to the
    broadband continuum. It then removes the forward model's finite-pixel,
    bandpass, spectral-bin, photon, etendue, exposure, port, and scalar gain
    factors. When the L1A contains detector-noise metadata, independent
    shot, read, and quantization variances are propagated through the same
    linear transform into the L1B ``radiance_noise`` field.
    """

    if not np.isfinite(noise_fraction) or noise_fraction < 0:
        msg = "noise_fraction must be finite and non-negative"
        raise ValueError(msg)
    if apodization is not None and (not np.isfinite(apodization) or apodization < 0):
        msg = "apodization must be None or a finite, non-negative Kaiser beta"
        raise ValueError(msg)

    ds = l1a.ds
    calibration = cal_db._ds
    if "image" not in ds or ds["image"].dims != ("pixelheight", "pixelcolumn"):
        msg = "L1A image must have pixelheight and pixelcolumn dimensions"
        raise ValueError(msg)
    if "sample_wavenumber" not in calibration:
        msg = "Calibration database must contain sample_wavenumber"
        raise ValueError(msg)

    image = np.asarray(ds["image"].to_numpy(), dtype=float)
    if np.any(~np.isfinite(image)):
        msg = "L1A image must be finite"
        raise ValueError(msg)
    if "opd_x" not in ds:
        msg = "L1A data must contain opd_x"
        raise ValueError(msg)
    opd_cm = np.asarray(ds["opd_x"].to_numpy(), dtype=float)
    if opd_cm.ndim != 1 or len(opd_cm) != image.shape[1]:
        msg = "opd_x must contain one value per detector column"
        raise ValueError(msg)
    opd_spacing_cm = np.diff(opd_cm)
    if (
        np.any(~np.isfinite(opd_cm))
        or np.any(opd_spacing_cm <= 0)
        or not np.allclose(opd_spacing_cm, opd_spacing_cm[0])
    ):
        msg = "opd_x must be finite, increasing, and uniformly spaced"
        raise ValueError(msg)

    sample_wavenumber = np.asarray(
        calibration["sample_wavenumber"].to_numpy(), dtype=float
    )
    if (
        sample_wavenumber.ndim != 1
        or len(sample_wavenumber) == 0
        or np.any(~np.isfinite(sample_wavenumber))
        or np.any(sample_wavenumber <= 0)
    ):
        msg = "sample_wavenumber must be a finite, positive one-dimensional grid"
        raise ValueError(msg)
    littrow_wavenumber = _scalar(ds, "Littrow")
    relative_wavenumber = sample_wavenumber - littrow_wavenumber
    wavenumber_spacing = _scalar(ds, "wavenumber_spacing")
    natural_spacing = 1 / (image.shape[1] * opd_spacing_cm[0])
    if not np.isclose(wavenumber_spacing, natural_spacing):
        msg = "L1A wavenumber spacing is inconsistent with its centered OPD grid"
        raise ValueError(msg)

    sample_indices = np.abs(relative_wavenumber) / wavenumber_spacing
    if not np.allclose(sample_indices, np.rint(sample_indices), atol=1e-8):
        msg = "Calibration samples must lie on the centered interferogram grid"
        raise ValueError(msg)

    inverse_operator = _inverse_cosine_operator(
        image.shape[1],
        opd_cm,
        relative_wavenumber,
        apodization,
    )
    modulation = np.einsum("yn,sn->ys", image, inverse_operator)
    detector_pixel_variance = _detector_pixel_variance(ds, image)
    if detector_pixel_variance is None:
        modulation_noise = None
    else:
        modulation_noise = np.sqrt(
            np.einsum(
                "yn,sn->ys",
                detector_pixel_variance,
                inverse_operator**2,
            )
        )

    wavelength_nm = 1e7 / sample_wavenumber
    wavelength_per_wavenumber = 1e7 / sample_wavenumber**2
    pixel_visibility = np.sinc(relative_wavenumber * _scalar(ds, "opd_pixel_width_cm"))
    if np.any(np.abs(pixel_visibility) <= np.finfo(float).eps):
        msg = "Finite-pixel visibility is zero at a calibration sample"
        raise ValueError(msg)

    denominator = (
        _filter_transmission(calibration, sample_wavenumber)
        * pixel_visibility
        * wavelength_per_wavenumber
        * wavenumber_spacing
        * _spectral_gain(ds, wavelength_nm)
    )
    radiance = modulation / denominator[np.newaxis, :]
    detector_radiance_noise = (
        None
        if modulation_noise is None
        else modulation_noise / np.abs(denominator)[np.newaxis, :]
    )

    spectral_order = np.argsort(sample_wavenumber)
    sorted_wavenumber = sample_wavenumber[spectral_order]
    if len(sorted_wavenumber) > 1:
        output_spacing = np.diff(sorted_wavenumber)
        if not np.allclose(output_spacing, output_spacing[0]):
            msg = "L1B data container requires uniformly spaced spectral samples"
            raise ValueError(msg)
        output_spacing = float(output_spacing[0])
    else:
        output_spacing = wavenumber_spacing

    radiance = radiance[:, spectral_order].T
    fractional_radiance_noise = np.abs(radiance) * noise_fraction
    if detector_radiance_noise is None:
        radiance_noise = fractional_radiance_noise
        noise_model = "fractional" if noise_fraction > 0 else "none"
    else:
        detector_radiance_noise = detector_radiance_noise[:, spectral_order].T
        radiance_noise = np.hypot(
            detector_radiance_noise,
            fractional_radiance_noise,
        )
        noise_model = "detector+fractional" if noise_fraction > 0 else "detector"

    saturated_fraction = _saturated_fraction(ds, image)
    if saturated_fraction is not None:
        radiance_noise[:, saturated_fraction > 0] = np.inf
    num_los = image.shape[0]
    if ds.sizes.get("los") != num_los:
        msg = "L1A geometry and detector image have different row counts"
        raise ValueError(msg)

    result = L1bImage.from_np_arrays(
        radiance=radiance,
        radiance_noise=radiance_noise,
        tangent_altitude=ds["tangent_altitude"].to_numpy(),
        tangent_latitude=ds["tangent_latitude"].to_numpy(),
        tangent_longitude=ds["tangent_longitude"].to_numpy(),
        left_wavenumber=np.full(num_los, sorted_wavenumber[0]),
        wavenumber_spacing=np.full(num_los, output_spacing),
        time=np.asarray(ds["time"].to_numpy()).reshape(-1)[0],
        observer_latitude=_scalar(ds, "spacecraft_latitude"),
        observer_longitude=_scalar(ds, "spacecraft_longitude"),
        observer_altitude=_scalar(ds, "spacecraft_altitude"),
        sza=ds["solar_zenith_angle"].to_numpy(),
        saa=ds["relative_solar_azimuth_angle"].to_numpy(),
        los_azimuth_angle=ds["los_azimuth_angle"].to_numpy(),
    )
    result.ds.attrs["apodization_window"] = (
        "rectangular" if apodization is None else "kaiser"
    )
    result.ds.attrs["radiance_noise_model"] = noise_model
    result.ds["radiance_noise"].attrs["description"] = (
        "One-sigma random-noise estimate; spectral covariance is not stored"
    )
    if saturated_fraction is not None:
        result.ds["saturated_fraction"] = xr.DataArray(
            saturated_fraction,
            dims=["los"],
        )
        result.ds["saturation_affected"] = xr.DataArray(
            saturated_fraction > 0,
            dims=["los"],
        )
    if apodization is not None:
        result.ds.attrs["apodization_beta"] = apodization
    return result
