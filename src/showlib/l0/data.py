from __future__ import annotations

from pathlib import Path

import xarray as xr


class L0Image:
    def __init__(self, ds: xr.Dataset):
        self._ds = ds

    @property
    def ds(self):
        return self._ds


class L0FileWriter:
    def __init__(self, l0_data: list[L0Image]) -> None:
        self._data = xr.concat([l0._ds for l0 in l0_data], dim="time")

    def _apply_global_attributes(self):
        pass

    def save(self, out_file: Path):
        self._data.to_netcdf(out_file.as_posix())


class L0DataSet:
    def __init__(self, file_path: Path):
        """
        Loads in a single L1a file and provides access to the data

        """
        self._ds = xr.open_dataset(file_path)
        self._name = file_path.stem
        self._parent = file_path.parent.parent

    @property
    def ds(self):
        return self._ds

    def image(self, sample: int):
        return L0Image(
            self._ds.isel(time=sample),
        )
