from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import resample

from .data import L1aImage


def coarse_zpd_determination(l1a: L1aImage):
    # An initial start is the maximum of the image
    l1a.ds["zpd"] = l1a.ds["image"].fillna(-999).argmax(dim="pixelcolumn", skipna=True)

    zpd_shift = np.zeros(len(l1a.ds["zpd"].to_numpy()))
    zpd_lmin = np.zeros(len(l1a.ds["zpd"].to_numpy()))
    zpd_rmin = np.zeros(len(l1a.ds["zpd"].to_numpy()))
    zpd_max = np.zeros(len(l1a.ds["zpd"].to_numpy()))
    zpd_lmin_v = np.zeros(len(l1a.ds["zpd"].to_numpy()))
    zpd_rmin_v = np.zeros(len(l1a.ds["zpd"].to_numpy()))
    zpd_max_v = np.zeros(len(l1a.ds["zpd"].to_numpy()))

    for i in range(len(zpd_shift)):
        zpd = l1a.ds["zpd"].to_numpy()[i]
        test = l1a.ds["image"].isel(heightrow=i).to_numpy()[zpd - 20 : zpd + 20]

        if zpd > 0:
            resampled = resample(test, len(test) * 100)

            maxidx = np.argmax(np.abs(resampled))
            sign = np.sign(resampled[maxidx])
            lminidx = np.argmin(sign * resampled[:maxidx])
            rminidx = np.argmin(sign * resampled[maxidx:])

            zpd_lmin[i] = -20 + lminidx / 100 + zpd
            zpd_rmin[i] = -20 + rminidx / 100 + zpd + maxidx
            zpd_max[i] = -20 + maxidx / 100 + zpd

            zpd_lmin_v[i] = resampled[lminidx]
            zpd_rmin_v[i] = resampled[rminidx + maxidx]
            zpd_max_v[i] = resampled[maxidx]
    # Find the ZPD between rows 10 and 50
    idx = [27, 120]

    true_zpd = interp1d(
        l1a.ds["heightrow"].to_numpy()[idx],
        (zpd_max[idx]),
        kind="linear",
        fill_value="extrapolate",
    )(l1a.ds["heightrow"].to_numpy())

    l1a.ds["zpd"] = (["heightrow"], true_zpd + l1a.ds["pixelcolumn"].to_numpy()[0])

    return l1a
