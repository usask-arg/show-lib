from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from showlib.l1b.data import L1bDataSet, L1bFileWriter, L1bImage

NUM_SAMPLES = 10
TANGENT_ALTITUDES = np.array([5000.0, 10000.0, 15000.0, 20000.0, 25000.0])
LEFT_WAVENUMBER = 7300.0
WAVENUMBER_SPACING = 0.5


def _l1b_image(radiance_value=2.0, t="2023-11-28T17:00:00") -> L1bImage:
    num_los = len(TANGENT_ALTITUDES)
    radiance = np.full((NUM_SAMPLES, num_los), radiance_value)
    return L1bImage.from_np_arrays(
        radiance=radiance,
        radiance_noise=radiance * 0.01,
        tangent_altitude=TANGENT_ALTITUDES,
        tangent_latitude=np.linspace(40, 41, num_los),
        tangent_longitude=np.linspace(-100, -99, num_los),
        left_wavenumber=np.full(num_los, LEFT_WAVENUMBER),
        wavenumber_spacing=np.full(num_los, WAVENUMBER_SPACING),
        time=pd.Timestamp(t),
        observer_latitude=39.0,
        observer_longitude=-101.0,
        observer_altitude=20000.0,
        sza=np.full(num_los, 60.0),
        saa=np.full(num_los, 90.0),
        los_azimuth_angle=np.zeros(num_los),
    )


def test_from_np_arrays_layout():
    ds = _l1b_image().ds

    assert ds["radiance"].dims == ("sample", "los")
    assert ds["radiance_noise"].dims == ("sample", "los")
    assert ds["tangent_altitude"].dims == ("los",)
    assert float(ds["spacecraft_altitude"]) == 20000.0


def test_skretrieval_l1_wavenumber_grid():
    l1 = _l1b_image().skretrieval_l1()["meas"].data

    expected = LEFT_WAVENUMBER + WAVENUMBER_SPACING * np.arange(NUM_SAMPLES)
    np.testing.assert_allclose(l1.wavenumber, expected)
    assert l1["radiance"].dims == ("wavenumber", "los")
    # Radiances that already look like W/m^2/nm/sr are passed through untouched
    np.testing.assert_array_equal(l1["radiance"], 2.0)
    np.testing.assert_array_equal(l1["radiance_noise"], 0.02)


def test_skretrieval_l1_filters():
    image = L1bImage(
        _l1b_image().ds,
        low_alt=7000,
        high_alt=22000,
        low_wvnumber_filter=7301,
        high_wvnumber_filter=7303,
    )
    l1 = image.skretrieval_l1()["meas"].data

    np.testing.assert_array_equal(l1.tangent_altitude, [10000.0, 15000.0, 20000.0])
    np.testing.assert_allclose(l1.wavenumber, [7301.5, 7302.0, 7302.5])
    assert l1["radiance"].shape == (3, 3)


def test_skretrieval_l1_converts_photon_units(caplog):
    photons = 1e12  # photons / s / cm^2 / sr / cm^-1
    with caplog.at_level(logging.WARNING):
        l1 = _l1b_image(radiance_value=photons).skretrieval_l1()["meas"].data

    assert "trying to convert" in caplog.text

    wvnum = l1.wavenumber.to_numpy()
    h = 6.62607015e-34
    c = 299792458
    joules_per_photon = h * c * wvnum * 100
    # per cm^2 -> per m^2, and per cm^-1 -> per nm
    expected = photons * joules_per_photon * 1e4 * wvnum**2 / 1e7

    np.testing.assert_allclose(l1["radiance"].isel(los=0), expected, rtol=1e-12)


def test_reference_values():
    image = _l1b_image()

    assert image.reference_latitude()["meas"] == pytest.approx(40.5)
    assert image.reference_longitude()["meas"] == pytest.approx(-99.5)
    # Solar zenith angles are stored in degrees
    assert image.reference_cos_sza()["meas"] == pytest.approx(0.5)


def test_sample_wavelengths():
    image = _l1b_image()

    wavelengths = image.sample_wavelengths()["meas"]
    np.testing.assert_allclose(
        wavelengths,
        1e7 / (LEFT_WAVENUMBER + WAVENUMBER_SPACING * np.arange(NUM_SAMPLES)),
    )


def test_append_information_to_l1():
    image = L1bImage(_l1b_image().ds, low_alt=7000, high_alt=22000)
    l1 = image.skretrieval_l1()
    l1["meas"].data = l1["meas"].data.drop_vars("tangent_altitude")

    image.append_information_to_l1(l1)

    np.testing.assert_array_equal(
        l1["meas"].data["tangent_altitude"], [10000.0, 15000.0, 20000.0]
    )


def test_dataset_image_selection():
    data = L1bDataSet.from_image(_l1b_image())
    assert data.ds.sizes["time"] == 1

    image = data.image(0, row_reduction=2, low_alt=0, high_alt=100000)

    assert isinstance(image, L1bImage)
    np.testing.assert_array_equal(image.ds.tangent_altitude, [5000.0, 15000.0, 25000.0])


def test_dataset_output_paths():
    parent = Path("/data/er2")
    data = L1bDataSet(
        xr.Dataset(), "HAWC_H2OL_Radiances_L1B_20231128T1700.v0_0_1.STD", parent
    )

    assert data.l2_path == parent / "l2" / (
        "HAWC_H2OL_Wvapor_L2_20231128T1700.v0_0_1.STD.nc"
    )
    assert data.l2_por_path == parent / "por" / (
        "HAWC_H2OL_Wvapor_L2_POR_20231128T1700.v0_0_1.STD.nc"
    )


def test_file_round_trip(tmp_path):
    l1b_folder = tmp_path / "l1b"
    l1b_folder.mkdir()
    out_file = l1b_folder / "HAWC_H2OL_Radiances_L1B_test.nc"

    L1bFileWriter(
        [_l1b_image(t="2023-11-28T17:00:00"), _l1b_image(t="2023-11-28T17:00:01")]
    ).save(out_file)
    data = L1bDataSet.from_file(out_file)

    assert data.ds.sizes["time"] == 2
    assert data.l2_path == tmp_path / "l2" / "HAWC_H2OL_Wvapor_L2_test.nc"
    np.testing.assert_array_equal(
        data.image(1).skretrieval_l1()["meas"].data["radiance"], 2.0
    )
