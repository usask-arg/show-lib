from __future__ import annotations

from pathlib import Path

import xarray as xr


class L1aImage:
    def __init__(self, ds: xr.Dataset):
        self._ds = ds

    @property
    def ds(self):
        return self._ds


class L1aFileWriter:
    def __init__(self, l1a_data: list[L1aImage]) -> None:
        self._data = xr.concat([l1a._ds for l1a in l1a_data], dim="time")

    def _apply_global_attributes(self):
        pass

    def save(self, out_file: Path):
        self._data.to_netcdf(out_file.as_posix())


class L1aDataSet:
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
        return L1aImage(
            self._ds.isel(time=sample),
        )
