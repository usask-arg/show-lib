from __future__ import annotations

import numpy as np

from showlib.l1a.data import L1aImage


def fill_nans(l1a: L1aImage, bad_nan_fraction=0.05):
    # First, any row with some fraction of NaN's should be marked as bad
    nan_fraction = np.isnan(l1a.ds["image"]).mean(dim="pixelcolumn")

    l1a.ds["image"] = l1a.ds["image"].where(nan_fraction < bad_nan_fraction)

    # TODO:
    # We should also mask rows with a nan near ZPD

    l1a.ds["image"] = l1a.ds["image"].interpolate_na(dim="pixelcolumn")

    return l1a
