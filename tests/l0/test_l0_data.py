from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from showlib.l0.data import L0DataSet, L0FileWriter, L0Image


def _l0_image(t: str, value: float) -> L0Image:
    ds = xr.Dataset({"image": (["pixelheight", "pixelcolumn"], np.full((3, 4), value))})
    ds.coords["time"] = pd.Timestamp(t)
    return L0Image(ds)


def test_l0_round_trip(tmp_path):
    images = [
        _l0_image("2023-11-28T17:00:00", 1.0),
        _l0_image("2023-11-28T17:00:01", 2.0),
    ]
    granule_folder = tmp_path / "l0"
    granule_folder.mkdir()
    out_file = granule_folder / "HAWC_H2OL_L0_test.nc"

    L0FileWriter(images).save(out_file)
    data = L0DataSet(out_file)

    assert data.ds.sizes == {"time": 2, "pixelheight": 3, "pixelcolumn": 4}
    assert data._name == "HAWC_H2OL_L0_test"
    assert data._parent == tmp_path

    second = data.image(1)
    assert isinstance(second, L0Image)
    np.testing.assert_array_equal(second.ds["image"].to_numpy(), 2.0)
