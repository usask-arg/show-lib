from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr


class L1AImage:
    def __init__(
        self,
        image: np.array,
        C2: np.array,
        noise: np.array,
        tangent_locations: np.array,
        tangent_longitudes: np.array,
        tangent_latitudes: np.array,
        spacecraft_latitude: np.array,
        spacecraft_longitude: np.array,
        spacecraft_altitude: np.array,
        solar_zenith_angle: np.array,
        relative_solar_azimuth_angle: np.array,
        los_azimuth_angle: np.array,
        time: datetime,
        filter_shape: np.array,
        abs_cal: float,
        pixel_response: np.array,
        bad_pixel_map: np.array,
        wavenumbers: np.array,
        wavenumber_spacing: float,
        Littrow: float,
        ThetaL: float,
        opd_x: np.array,
        pos_x: np.array,
        pos_y: np.array,
    ):
        self._ds = xr.Dataset()

        self._ds["image"] = xr.DataArray(image, dims=["pixelheight", "pixelcolumn"])
        self._ds["noise"] = xr.DataArray(noise, dims=["pixelheight", "pixelcolumn"])
        self._ds["C2"] = xr.DataArray(C2, dims=["los"])

        # Merge the geometry information with the L0 file
        self._ds["tangent_altitude"] = xr.DataArray(tangent_locations, dims=["los"])
        self._ds["tangent_latitude"] = xr.DataArray(tangent_latitudes, dims=["los"])
        self._ds["tangent_longitude"] = xr.DataArray(tangent_longitudes, dims=["los"])
        self._ds["spacecraft_latitude"] = spacecraft_latitude
        self._ds["spacecraft_longitude"] = spacecraft_longitude
        self._ds["spacecraft_altitude"] = spacecraft_altitude
        self._ds["solar_zenith_angle"] = xr.DataArray(solar_zenith_angle, dims=["los"])
        self._ds["relative_solar_azimuth_angle"] = xr.DataArray(
            relative_solar_azimuth_angle, dims=["los"]
        )
        self._ds["los_azimuth_angle"] = xr.DataArray(los_azimuth_angle, dims=["los"])
        self._ds["time"] = time
        self._ds["filter_shape"] = filter_shape
        self._ds["pixel_response"] = pixel_response
        self._ds["abs_cal"] = abs_cal
        self._ds["wavenumbers"] = wavenumbers
        self._ds["bad_pixel_map"] = xr.DataArray(
            bad_pixel_map, dims=["pixelheight", "pixelcolumn"]
        )
        self._ds["wavenumber_spacing"] = wavenumber_spacing
        self._ds["Littrow"] = Littrow
        self._ds["ThetaL"] = ThetaL
        self._ds["opd_x"] = opd_x
        self._ds["pos_x"] = pos_x
        self._ds["pos_y"] = pos_y

    @property
    def ds(self):
        return self._ds


class L1AFileWriter:
    def __init__(self, L1A_images: list[L1AImage]) -> None:
        self._data = xr.concat([l1a_image.ds for l1a_image in L1A_images], dim="time")

        self._apply_global_attributes()

    def _apply_global_attributes(self):
        pass

    def save(self, out_file: Path):
        self._data.to_netcdf(out_file.as_posix())
