from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from showlib.l1a.data import L1ADataSet, L1AImage
from showlib.schemas import SchemaValidationError

NUM_ROWS = 5
NUM_COLS = 8


def _l1a_image(t: str, exposure_time=0.1, **overrides) -> L1AImage:
    rows = np.arange(NUM_ROWS, dtype=float)
    fields = {
        "image": np.full((NUM_ROWS, NUM_COLS), 1000.0),
        "time": pd.Timestamp(t),
        "tangent_altitude": 10000 + 1000 * rows,
        "tangent_latitude": 40 + 0.1 * rows,
        "tangent_longitude": -100 + 0.1 * rows,
        "spacecraft_latitude": 39.0,
        "spacecraft_longitude": -101.0,
        "spacecraft_altitude": 20000.0,
        "solar_zenith_angle": np.full(NUM_ROWS, 60.0),
        "relative_solar_azimuth_angle": np.full(NUM_ROWS, 90.0),
        "los_azimuth_angle": np.zeros(NUM_ROWS),
        "exposure_time": exposure_time,
    }
    return L1AImage(**{**fields, **overrides})


def _images():
    return [_l1a_image("2023-11-28T17:00:00"), _l1a_image("2023-11-28T17:00:01")]


def _simulation():
    return xr.Dataset(
        {
            "image_noiseless": (
                ["time", "los", "pixelcolumn"],
                np.full((2, NUM_ROWS, NUM_COLS), 1000.0),
            ),
            "noise_seed": ((), np.int64(3)),
        }
    )


def test_image_layout():
    ds = _l1a_image("2023-11-28T17:00:00").ds

    assert ds["image"].dims == ("los", "pixelcolumn")
    assert ds["tangent_altitude"].dims == ("los",)
    assert ds["exposure_time"].dims == ()
    assert int(ds["num_averaged_exposures"]) == 1
    assert "time" in ds.coords


def test_image_wraps_azimuths():
    ds = _l1a_image(
        "2023-11-28T17:00:00",
        relative_solar_azimuth_angle=np.full(NUM_ROWS, 270.0),
        los_azimuth_angle=np.full(NUM_ROWS, -90.0),
    ).ds

    np.testing.assert_allclose(ds["relative_solar_azimuth_angle"], -90.0)
    np.testing.assert_allclose(ds["los_azimuth_angle"], 270.0)


def test_granule_stacks_images_along_time():
    granule = L1ADataSet.from_images(_images())

    assert granule.ds["image"].dims == ("time", "los", "pixelcolumn")
    assert granule.ds["spacecraft_altitude"].dims == ("time",)
    assert granule.ds.sizes["time"] == 2
    assert granule.ds["image"].attrs["units"] == "count"
    assert "schema_version" in granule.ds.attrs
    assert granule.simulation is None


def test_granule_rejects_invalid_images():
    with pytest.raises(SchemaValidationError, match="exposure_time"):
        L1ADataSet.from_images([_l1a_image("2023-11-28T17:00:00", exposure_time=-1.0)])


def test_granule_rejects_images_out_of_time_order():
    with pytest.raises(SchemaValidationError, match="time-increasing"):
        L1ADataSet.from_images(_images()[::-1])


def test_granule_round_trip(tmp_path):
    path = tmp_path / "l1a.nc"
    L1ADataSet.from_images(_images()).save(path)

    granule = L1ADataSet.from_file(path)

    assert granule.ds.sizes["time"] == 2
    assert granule.simulation is None
    np.testing.assert_allclose(
        granule.ds["tangent_altitude"].isel(time=0), 10000 + 1000 * np.arange(NUM_ROWS)
    )


def test_granule_round_trip_with_simulation_group(tmp_path):
    path = tmp_path / "l1a.nc"
    L1ADataSet.from_images(_images(), simulation=_simulation()).save(path)

    granule = L1ADataSet.from_file(path)

    assert int(granule.simulation["noise_seed"]) == 3
    assert granule.simulation["image_noiseless"].attrs["units"] == "count"
    assert "image_noiseless" not in granule.ds
