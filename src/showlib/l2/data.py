from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr


class L2Profile:
    def __init__(
        self,
        altitude_m: np.array,
        h2o_vmr: np.array,
        latitude: float,
        longitude: float,
        time: datetime,
    ):
        """
        Defines a single L2 profile for the SHOW instrument

        Parameters
        ----------
        altitude_m: np.array
            Altitudes in [m] above the egm96 surface. Shape (n,)

        h2o_vmr: np.array
            Water vapor volume mixing ratio on the same grid as altitude_m. Shape (n,)

        latitude: float
            Latitude of the profile in [degrees north]

        longitude: float
            Longitude of the profile in [degrees east]. Will be converted to the range [-180, 180] if not in this range.

        time: datetime
            UTC time for the profile.
        """
        self._ds = xr.Dataset()

        self._ds["h2o_vmr"] = xr.DataArray(h2o_vmr, dims=["altitude"])

        self._ds.coords["altitude"] = altitude_m
        self._ds.coords["time"] = time
        self._ds["latitude"] = latitude
        self._ds["longitude"] = longitude


class L2FileWriter:
    def __init__(self, l2_profiles: list[L2Profile]) -> None:
        self._data = xr.concat(l2_profiles, dim="time")

        self._apply_global_attributes()

    def _apply_global_attributes(self):
        pass

    def save(self, out_file: Path):
        self._data.to_netcdf(out_file.as_posix())
