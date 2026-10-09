from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from showlib.l1a.data import L1AFileWriter, L1AImage

NUM_ROWS = 5
NUM_COLS = 8


def _l1a_image(t: str) -> L1AImage:
    los = np.arange(NUM_ROWS, dtype=float)
    return L1AImage(
        image=np.ones((NUM_ROWS, NUM_COLS)),
        tangent_locations=10000 + 1000 * los,
        tangent_longitudes=-100 + 0.1 * los,
        tangent_latitudes=40 + 0.1 * los,
        spacecraft_latitude=39.0,
        spacecraft_longitude=-101.0,
        spacecraft_altitude=20000.0,
        solar_zenith_angle=np.full(NUM_ROWS, 60.0),
        relative_solar_azimuth_angle=np.full(NUM_ROWS, 90.0),
        los_azimuth_angle=np.zeros(NUM_ROWS),
        time=pd.Timestamp(t),
    )


def test_l1a_image_layout():
    ds = _l1a_image("2023-11-28T17:00:00").ds

    assert ds["image"].dims == ("pixelheight", "pixelcolumn")
    for var in [
        "tangent_altitude",
        "tangent_latitude",
        "tangent_longitude",
        "solar_zenith_angle",
        "relative_solar_azimuth_angle",
        "los_azimuth_angle",
    ]:
        assert ds[var].dims == ("los",)
    assert float(ds["spacecraft_altitude"]) == 20000.0


def test_l1a_round_trip(tmp_path):
    images = [
        _l1a_image("2023-11-28T17:00:00"),
        _l1a_image("2023-11-28T17:00:01"),
    ]
    out_file = tmp_path / "l1a.nc"
    L1AFileWriter(images).save(out_file)

    with xr.open_dataset(out_file) as ds:
        assert ds["image"].dims == ("time", "pixelheight", "pixelcolumn")
        assert ds.sizes["time"] == 2
        np.testing.assert_allclose(
            ds["tangent_altitude"].isel(time=0), 10000 + 1000 * np.arange(NUM_ROWS)
        )
