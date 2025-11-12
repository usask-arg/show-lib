from __future__ import annotations

import numpy as np
import xarray as xr


class CalibrationDatabase:
    def __init__(self, ds: xr.Dataset):
        self._ds = ds

    def bad_pixel_map(self, time) -> np.ndarray:
        pass

    def dark_current(self, time, detector_temperature) -> np.ndarray:
        pass

    def nominal_central_wavenumbers(self, time) -> np.ndarray:  # noqa: ARG002
        return self._ds["sample_wavenumber"].to_numpy()
