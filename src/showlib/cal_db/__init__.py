import xarray as xr
import numpy as np


class CalibrationDatabase:
    def __init__(self, ds: xr.Dataset):
        self._ds = ds

    def bad_pixel_map(self, time) -> np.ndarray:
        pass

    def dark_current(self, time, detector_temperature) -> np.ndarray:
        pass

    def nominal_central_wavenumbers(self, time) -> np.ndarray:
        return self._ds["sample_wavenumber"].to_numpy()


    @classmethod
    def from_np_arrays(

    ):
        ds = xr.Dataset()

        ds["filter_shape"] = filter_shape
        ds["pixel_response"] = pixel_response
        ds["abs_cal"] = abs_cal
        ds["wavenumbers"] = wavenumbers
        ds["bad_pixel_map"] = xr.DataArray(
            bad_pixel_map, dims=["pixelheight", "pixelcolumn"]
        )
        ds["wavenumber_spacing"] = wavenumber_spacing
        ds["Littrow"] = Littrow
        ds["ThetaL"] = ThetaL
        ds["opd_x"] = opd_x
        ds["pos_x"] = pos_x
        ds["pos_y"] = pos_y
        ds["zpd"] = zpd
