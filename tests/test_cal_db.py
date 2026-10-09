from __future__ import annotations

import numpy as np
import xarray as xr

from showlib.cal_db import CalibrationDatabase


def test_nominal_central_wavenumbers():
    wavenumbers = np.linspace(7295, 7340, 10)
    ds = xr.Dataset(coords={"sample_wavenumber": wavenumbers})

    cal_db = CalibrationDatabase(ds)

    np.testing.assert_array_equal(cal_db.nominal_central_wavenumbers(None), wavenumbers)
