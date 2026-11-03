from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from showlib.l1b.data import L1bDataSet, L1bImage
from showlib.schemas import SchemaValidationError

NUM_SAMPLES = 10
TANGENT_ALTITUDES = np.array([5000.0, 10000.0, 15000.0, 20000.0, 25000.0])
NUM_LOS = len(TANGENT_ALTITUDES)
WAVENUMBER = 7300.0 + 0.5 * np.arange(NUM_SAMPLES)
PROCESSING = {"apodization_window": "rectangular", "radiance_noise_model": "detector"}


def _l1b_image(radiance_value=2.0, t="2023-11-28T17:00:00", **overrides) -> L1bImage:
    radiance = np.full((NUM_LOS, NUM_SAMPLES), radiance_value)
    fields = {
        "radiance": radiance,
        "radiance_noise": radiance * 0.01,
        "wavenumber": WAVENUMBER,
        "tangent_altitude": TANGENT_ALTITUDES,
        "tangent_latitude": np.linspace(40, 41, NUM_LOS),
        "tangent_longitude": np.linspace(-100, -99, NUM_LOS),
        "time": pd.Timestamp(t),
        "spacecraft_latitude": 39.0,
        "spacecraft_longitude": -101.0,
        "spacecraft_altitude": 20000.0,
        "solar_zenith_angle": np.full(NUM_LOS, 60.0),
        "relative_solar_azimuth_angle": np.full(NUM_LOS, 90.0),
        "los_azimuth_angle": np.zeros(NUM_LOS),
        "processing": PROCESSING,
    }
    return L1bImage.from_np_arrays(**{**fields, **overrides})


def _granule(**overrides) -> L1bDataSet:
    return L1bDataSet.from_images(
        [
            _l1b_image(t="2023-11-28T17:00:00", **overrides),
            _l1b_image(t="2023-11-28T17:00:01", **overrides),
        ]
    )


def test_from_np_arrays_layout():
    ds = _l1b_image().ds

    assert ds["radiance"].dims == ("los", "spectral_sample")
    assert ds["radiance_noise"].dims == ("los", "spectral_sample")
    assert ds["wavenumber"].dims == ("los", "spectral_sample")
    assert ds["tangent_altitude"].dims == ("los",)
    assert float(ds["spacecraft_altitude"]) == 20000.0
    np.testing.assert_array_equal(ds["saturated_fraction"], 0.0)
    assert ds.attrs == PROCESSING


def test_skretrieval_l1_wavenumber_grid():
    l1 = _l1b_image().skretrieval_l1()["meas"].data

    np.testing.assert_allclose(l1.wavenumber, WAVENUMBER)
    assert l1["radiance"].dims == ("wavenumber", "los")
    np.testing.assert_array_equal(l1["radiance"], 2.0)
    np.testing.assert_array_equal(l1["radiance_noise"], 0.02)


def test_skretrieval_l1_passes_large_radiances_through_unchanged():
    l1 = _l1b_image(radiance_value=1e12).skretrieval_l1()["meas"].data
    np.testing.assert_array_equal(l1["radiance"], 1e12)


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


def test_skretrieval_l1_rejects_different_grids_per_los():
    wavenumber = np.broadcast_to(WAVENUMBER, (NUM_LOS, NUM_SAMPLES)).copy()
    wavenumber[2] += 0.1
    image = _l1b_image(wavenumber=wavenumber)

    with pytest.raises(NotImplementedError, match="wavenumber grid"):
        image.skretrieval_l1()


def test_saturated_rows_are_excluded_from_the_retrieval():
    saturated_fraction = np.zeros(NUM_LOS)
    saturated_fraction[1] = 0.25
    image = _l1b_image(saturated_fraction=saturated_fraction)

    l1 = image.skretrieval_l1()["meas"].data

    assert 10000.0 not in l1.tangent_altitude
    assert l1.sizes["los"] == NUM_LOS - 1
    assert len(image.sk2_geometry()["meas"].observer_rays) == NUM_LOS - 1


def test_all_rows_saturated_is_an_error():
    image = _l1b_image(saturated_fraction=np.ones(NUM_LOS))
    with pytest.raises(ValueError, match="No unsaturated L1B rows"):
        image.skretrieval_l1()


def test_reference_values():
    image = _l1b_image()

    assert image.reference_latitude()["meas"] == pytest.approx(40.5)
    assert image.reference_longitude()["meas"] == pytest.approx(-99.5)
    assert image.reference_cos_sza()["meas"] == pytest.approx(0.5)


def test_sample_wavelengths():
    wavelengths = _l1b_image().sample_wavelengths()["meas"]
    np.testing.assert_allclose(wavelengths, 1e7 / WAVENUMBER)


def test_append_information_to_l1():
    image = L1bImage(_l1b_image().ds, low_alt=7000, high_alt=22000)
    l1 = image.skretrieval_l1()
    l1["meas"].data = l1["meas"].data.drop_vars("tangent_altitude")

    image.append_information_to_l1(l1)

    np.testing.assert_array_equal(
        l1["meas"].data["tangent_altitude"], [10000.0, 15000.0, 20000.0]
    )


def test_granule_layout():
    granule = _granule()

    assert granule.ds["radiance"].dims == ("time", "los", "spectral_sample")
    assert granule.ds["wavenumber"].dims == ("los", "spectral_sample")
    assert granule.ds["radiance"].attrs["units"] == "W m-2 sr-1 nm-1"
    assert granule.ds.attrs["apodization_window"] == "rectangular"


def test_granule_image_selection():
    image = _granule().image(1, row_reduction=2, low_alt=0, high_alt=100000)

    assert isinstance(image, L1bImage)
    np.testing.assert_array_equal(image.ds.tangent_altitude, [5000.0, 15000.0, 25000.0])
    assert image.ds["time"] == np.datetime64("2023-11-28T17:00:01")


def test_granule_rejects_different_wavenumber_grids():
    images = [
        _l1b_image(t="2023-11-28T17:00:00"),
        _l1b_image(t="2023-11-28T17:00:01", wavenumber=WAVENUMBER + 0.1),
    ]
    with pytest.raises(ValueError, match="wavenumber grid"):
        L1bDataSet.from_images(images)


def test_granule_rejects_different_processing_settings():
    images = [
        _l1b_image(t="2023-11-28T17:00:00"),
        _l1b_image(
            t="2023-11-28T17:00:01",
            processing={**PROCESSING, "radiance_noise_model": "fractional"},
        ),
    ]
    with pytest.raises(ValueError, match="processing settings"):
        L1bDataSet.from_images(images)


def test_granule_rejects_invalid_images():
    with pytest.raises(SchemaValidationError, match="apodization_window"):
        _granule(processing={"radiance_noise_model": "detector"})


def test_granule_output_paths():
    parent = Path("/data/er2")
    granule = L1bDataSet(
        _granule().ds, "HAWC_H2OL_Radiances_L1B_20231128T1700.v0_0_1.STD", parent
    )

    assert granule.l2_path == parent / "l2" / (
        "HAWC_H2OL_Wvapor_L2_20231128T1700.v0_0_1.STD.nc"
    )
    assert granule.l2_por_path == parent / "por" / (
        "HAWC_H2OL_Wvapor_L2_POR_20231128T1700.v0_0_1.STD.nc"
    )


def test_granule_round_trip(tmp_path):
    l1b_folder = tmp_path / "l1b"
    l1b_folder.mkdir()
    out_file = l1b_folder / "HAWC_H2OL_Radiances_L1B_test.nc"

    _granule().save(out_file)
    granule = L1bDataSet.from_file(out_file)

    assert granule.ds.sizes["time"] == 2
    assert granule.l2_path == tmp_path / "l2" / "HAWC_H2OL_Wvapor_L2_test.nc"
    np.testing.assert_array_equal(
        granule.image(1).skretrieval_l1()["meas"].data["radiance"], 2.0
    )
