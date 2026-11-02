from __future__ import annotations

import numpy as np
import xarray as xr

from showlib.cal_db import CalibrationDatabase
from showlib.l1a.data import L1AImage
from showlib.processing.l1a_to_l1b import (
    _filter_transmission,
    _inverse_cosine_operator,
    _inverse_cosine_transform,
    _spectral_gain,
    process_l1a_to_l1b_dataset,
)


def _simple_l1a_and_calibration():
    num_columns = 16
    opd_spacing_cm = 0.1
    opd_pixel_width_cm = 0.08
    littrow = 1000.0
    wavenumber_spacing = 1 / (num_columns * opd_spacing_cm)
    sample_wavenumber = littrow - np.arange(1, 4) * wavenumber_spacing
    relative_wavenumber = sample_wavenumber - littrow
    opd = (np.arange(num_columns) - (num_columns - 1) / 2) * opd_spacing_cm

    filter_transmission = np.array([0.75, 0.85, 0.95])
    calibration_ds = xr.Dataset(
        {
            "sample_wavenumber": ("sample_wavenumber", sample_wavenumber),
            "filter": ("hires_wavenumber", filter_transmission),
        },
        coords={"hires_wavenumber": sample_wavenumber},
    ).sortby("hires_wavenumber")

    expected_radiance = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    wavelength_nm = 1e7 / sample_wavenumber
    wavelength_per_wavenumber = 1e7 / sample_wavenumber**2
    column_etendue_m2_sr = 1.5e-10
    exposure_time_s = 2.0
    port_fraction = 0.5
    system_efficiency = 0.8
    scalar_gain = 1.2
    photon_energy_j = 6.62607015e-34 * 299792458.0 / (wavelength_nm * 1e-9)
    spectral_gain = (
        scalar_gain
        * column_etendue_m2_sr
        * exposure_time_s
        * port_fraction
        * system_efficiency
        / photon_energy_j
    )
    modulation = (
        expected_radiance
        * filter_transmission[np.newaxis, :]
        * np.sinc(relative_wavenumber * opd_pixel_width_cm)[np.newaxis, :]
        * wavelength_per_wavenumber[np.newaxis, :]
        * wavenumber_spacing
        * spectral_gain[np.newaxis, :]
    )
    cosine_basis = np.cos(
        2 * np.pi * relative_wavenumber[:, np.newaxis] * opd[np.newaxis, :]
    )
    image = 1000.0 + modulation @ cosine_basis

    l1a = L1AImage(
        image=image,
        tangent_locations=np.array([10000.0, 11000.0]),
        tangent_longitudes=np.array([-106.0, -106.0]),
        tangent_latitudes=np.array([52.0, 52.0]),
        spacecraft_latitude=53.0,
        spacecraft_longitude=-105.0,
        spacecraft_altitude=450000.0,
        solar_zenith_angle=np.array([60.0, 60.0]),
        relative_solar_azimuth_angle=np.array([25.0, 25.0]),
        los_azimuth_angle=np.array([5.0, 5.0]),
        time=np.datetime64("2021-01-01T12:00:00"),
    )
    l1a.ds["opd_x"] = ("pixelcolumn", opd)
    l1a.ds["opd_pixel_width_cm"] = opd_pixel_width_cm
    l1a.ds["Littrow"] = littrow
    l1a.ds["wavenumber_spacing"] = wavenumber_spacing
    l1a.ds["interferogram_gain"] = scalar_gain
    l1a.ds["column_etendue_m2_sr"] = column_etendue_m2_sr
    l1a.ds["exposure_time_s"] = exposure_time_s
    l1a.ds["interferometer_port_fraction"] = port_fraction
    l1a.ds["system_efficiency"] = system_efficiency

    return l1a, CalibrationDatabase(calibration_ds), expected_radiance


def test_simple_inverse_recovers_input_radiance():
    l1a, calibration, expected_radiance = _simple_l1a_and_calibration()

    result = process_l1a_to_l1b_dataset(
        l1a,
        calibration,
        apodization=None,
    )

    spectral_order = np.argsort(calibration._ds["sample_wavenumber"].to_numpy())
    np.testing.assert_allclose(
        result.ds["radiance"],
        expected_radiance[:, spectral_order].T,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(result.ds["radiance_noise"], 0.0)
    np.testing.assert_allclose(
        result.ds["left_wavenumber"],
        calibration._ds["sample_wavenumber"].min(),
    )
    assert result.ds.attrs["apodization_window"] == "rectangular"


def test_kaiser_apodization_is_enabled_by_default():
    l1a, calibration, _expected_radiance = _simple_l1a_and_calibration()

    default_result = process_l1a_to_l1b_dataset(l1a, calibration)
    explicit_result = process_l1a_to_l1b_dataset(
        l1a,
        calibration,
        apodization=10.0,
    )
    unapodized_result = process_l1a_to_l1b_dataset(
        l1a,
        calibration,
        apodization=None,
    )

    np.testing.assert_allclose(
        default_result.ds["radiance"],
        explicit_result.ds["radiance"],
    )
    assert not np.allclose(
        default_result.ds["radiance"],
        unapodized_result.ds["radiance"],
    )
    assert default_result.ds.attrs["apodization_window"] == "kaiser"
    assert default_result.ds.attrs["apodization_beta"] == 10.0


def test_invalid_apodization_is_rejected():
    l1a, calibration, _expected_radiance = _simple_l1a_and_calibration()

    with np.testing.assert_raises_regex(ValueError, "apodization"):
        process_l1a_to_l1b_dataset(l1a, calibration, apodization=-1.0)


def test_apodization_uses_unapodized_transform_normalization():
    num_columns = 16
    opd_spacing_cm = 0.1
    opd = (np.arange(num_columns) - (num_columns - 1) / 2) * opd_spacing_cm
    relative_wavenumber = np.array([-1 / (num_columns * opd_spacing_cm)])
    basis = np.cos(2 * np.pi * relative_wavenumber[:, np.newaxis] * opd)
    line_amplitude = 3.0
    image = 1000.0 + line_amplitude * basis

    result = _inverse_cosine_transform(
        image,
        opd,
        relative_wavenumber,
        apodization=10.0,
    )

    window = np.kaiser(num_columns, 10.0)
    expected = line_amplitude * np.sum(window * basis[0] ** 2) / np.sum(basis[0] ** 2)
    np.testing.assert_allclose(result[0, 0], expected)
    assert result[0, 0] < line_amplitude


def test_noise_fraction_scales_recovered_radiance():
    l1a, calibration, _expected_radiance = _simple_l1a_and_calibration()

    result = process_l1a_to_l1b_dataset(l1a, calibration, noise_fraction=0.01)

    np.testing.assert_allclose(
        result.ds["radiance_noise"], np.abs(result.ds["radiance"]) * 0.01
    )


def test_detector_noise_is_propagated_through_inverse_transform():
    l1a, calibration, _expected_radiance = _simple_l1a_and_calibration()
    expected_electrons = xr.full_like(l1a.ds["image"], 10_000.0)
    read_noise_electrons = 30.0
    electrons_per_dn = 100.0
    num_averaged_exposures = 4
    l1a.ds["noise_enabled"] = False
    l1a.ds["detector_noise_estimate_available"] = True
    l1a.ds["image_noiseless"] = expected_electrons
    l1a.ds["detector_read_noise_electrons"] = read_noise_electrons
    l1a.ds["detector_electrons_per_dn"] = electrons_per_dn
    l1a.ds["num_averaged_exposures"] = num_averaged_exposures
    l1a.ds["saturated"] = xr.zeros_like(l1a.ds["image"], dtype=bool)

    result = process_l1a_to_l1b_dataset(
        l1a,
        calibration,
        apodization=None,
    )

    sample_wavenumber = calibration._ds["sample_wavenumber"].to_numpy()
    relative_wavenumber = sample_wavenumber - l1a.ds["Littrow"].item()
    inverse_operator = _inverse_cosine_operator(
        l1a.ds.sizes["pixelcolumn"],
        l1a.ds["opd_x"].to_numpy(),
        relative_wavenumber,
        None,
    )
    wavelength_nm = 1e7 / sample_wavenumber
    denominator = (
        _filter_transmission(calibration._ds, sample_wavenumber)
        * np.sinc(relative_wavenumber * l1a.ds["opd_pixel_width_cm"].item())
        * 1e7
        / sample_wavenumber**2
        * l1a.ds["wavenumber_spacing"].item()
        * _spectral_gain(l1a.ds, wavelength_nm)
    )
    pixel_variance = (
        10_000.0 + read_noise_electrons**2 + electrons_per_dn**2 / 12
    ) / num_averaged_exposures
    expected_noise = np.sqrt(
        pixel_variance * np.sum(inverse_operator**2, axis=1)
    ) / np.abs(denominator)
    spectral_order = np.argsort(sample_wavenumber)
    expected_noise = np.repeat(
        expected_noise[spectral_order, np.newaxis],
        l1a.ds.sizes["pixelheight"],
        axis=1,
    )

    np.testing.assert_allclose(result.ds["radiance_noise"], expected_noise)
    assert result.ds.attrs["radiance_noise_model"] == "detector"
    assert not result.ds["saturation_affected"].to_numpy().any()


def test_saturation_marks_l1b_row_as_unusable():
    l1a, calibration, _expected_radiance = _simple_l1a_and_calibration()
    l1a.ds["noise_enabled"] = True
    l1a.ds["image_noiseless"] = xr.full_like(l1a.ds["image"], 10_000.0)
    l1a.ds["detector_read_noise_electrons"] = 30.0
    l1a.ds["detector_electrons_per_dn"] = 100.0
    saturated = np.zeros_like(l1a.ds["image"], dtype=bool)
    saturated[0, 3] = True
    l1a.ds["saturated"] = xr.DataArray(
        saturated,
        dims=l1a.ds["image"].dims,
    )

    result = process_l1a_to_l1b_dataset(l1a, calibration)

    assert np.isinf(result.ds["radiance_noise"].isel(los=0)).all()
    assert np.isfinite(result.ds["radiance_noise"].isel(los=1)).all()
    assert result.ds["saturation_affected"].to_numpy().tolist() == [True, False]
    np.testing.assert_allclose(
        result.ds["saturated_fraction"],
        [1 / l1a.ds.sizes["pixelcolumn"], 0.0],
    )

    retrieval_l1 = result.skretrieval_l1()["meas"].data
    assert retrieval_l1.sizes["los"] == 1
    np.testing.assert_allclose(retrieval_l1["tangent_altitude"], [11_000.0])
    assert len(result.sk2_geometry()["meas"].observer_rays) == 1

    result.ds["saturation_affected"] = xr.ones_like(
        result.ds["saturation_affected"], dtype=bool
    )
    with np.testing.assert_raises_regex(ValueError, "No unsaturated L1B rows"):
        result.skretrieval_l1()
