from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from showlib.l2.data import L2FileWriter, L2Profile

ALTITUDES = np.arange(5000.0, 30001.0, 5000.0)


def _profile(t: str, scale: float = 1.0) -> L2Profile:
    n = len(ALTITUDES)
    return L2Profile(
        altitude_m=ALTITUDES,
        h2o_vmr=np.full(n, 5e-6) * scale,
        h2o_vmr_1sigma=np.full(n, 1e-7),
        h2o_vmr_prior=np.full(n, 4e-6),
        tropopause_altitude=11500.0,
        averaging_kernel=np.eye(n),
        latitude=40.0,
        longitude=-100.0,
        time=pd.Timestamp(t),
    )


def test_l2_data_construction():
    ds = _profile("2023-11-28T17:00:00").ds

    for var in ["h2o_vmr", "h2o_vmr_1sigma", "h2o_vmr_prior"]:
        assert ds[var].dims == ("altitude",)
    assert ds["averaging_kernel"].dims == ("altitude", "altitude2")
    np.testing.assert_array_equal(ds.altitude, ALTITUDES)
    np.testing.assert_array_equal(ds.altitude2, ALTITUDES)
    assert float(ds["tropopause_altitude"]) == 11500.0
    assert float(ds["latitude"]) == 40.0
    assert float(ds["longitude"]) == -100.0


def test_l2_file_round_trip(tmp_path):
    profiles = [
        _profile("2023-11-28T17:00:00", scale=1.0),
        _profile("2023-11-28T17:01:00", scale=2.0),
    ]
    out_file = tmp_path / "l2.nc"
    L2FileWriter(profiles).save(out_file)

    with xr.open_dataset(out_file) as ds:
        assert ds.sizes == {
            "time": 2,
            "altitude": len(ALTITUDES),
            "altitude2": len(ALTITUDES),
        }
        assert ds["h2o_vmr"].dims == ("time", "altitude")
        np.testing.assert_allclose(ds["h2o_vmr"].isel(time=1), 1e-5)
        assert pd.Timestamp(ds.time.to_numpy()[1]) == pd.Timestamp(
            "2023-11-28T17:01:00"
        )
