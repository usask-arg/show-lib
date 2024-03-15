from __future__ import annotations

import xarray as xr

from showlib.l1a.data import L1aImage
from showlib.l1a.dcbias import add_dcbias
from showlib.l1a.fill import fill_nans
from showlib.l1a.source_variation import correct_source_variation
from showlib.l1a.zpd import coarse_zpd_determination


def process_l1a_to_l1b(l1a: L1aImage):
    # Processing steps
    l1a = fill_nans(l1a)

    # Start with a coarse determination of ZPD
    l1a = coarse_zpd_determination(l1a)

    # Add on the DC bias to the image
    l1a = add_dcbias(l1a)

    # To correct the source variation
    l1a = correct_source_variation(l1a)


if __name__ == "__main__":
    l1a = L1aImage(xr.open_dataset("/Users/dannyz/Documents/l1a_image_1794.nc"))
    process_l1a_to_l1b(l1a)
