from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import interpolate

from showlib.l1a.data import L1aImage


def correct_source_variation(l1a: L1aImage):
    n = l1a.ds["image"].shape[1]

    spectrum_l1 = np.fft.rfft(l1a.ds["image"].to_numpy(), axis=-1)

    s = 150
    n = 80

    f = ((1 + np.cos(np.pi * np.arange(0, spectrum_l1.shape[1]) / s)) / 2) ** n

    interf_filter = np.fft.irfft(spectrum_l1 * f[np.newaxis, :])

    filter_zpd = np.zeros(len(l1a.ds["heightrow"]))

    for i in range(len(filter_zpd)):
        filter_zpd[i] = interpolate.interp1d(
            l1a.ds["pixelcolumn"].to_numpy(),
            interf_filter[i],
            kind="linear",
            fill_value="extrapolate",
        )(l1a.ds["zpd"].to_numpy()[i])

    interf_filter = xr.DataArray(
        interf_filter,
        dims=["heightrow", "pixelcolumn"],
        coords={"heightrow": l1a.ds["heightrow"], "pixelcolumn": l1a.ds["pixelcolumn"]},
    )
    interf_filter = interf_filter.rolling(
        pixelcolumn=5, center=True, min_periods=1
    ).mean()

    l1a.ds["image"] /= interf_filter
    l1a.ds["image"] *= filter_zpd[:, np.newaxis]

    l1a.ds["image"] -= l1a.ds["image"].mean(dim="pixelcolumn")
