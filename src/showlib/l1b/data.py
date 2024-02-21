from __future__ import annotations

import abc
from pathlib import Path

import sasktran2 as sk
import xarray as xr


class L1bDataSet:
    def __init__(self, file_path: Path):
        """
        Loads in a single L1b file and provides access to the data

        """
        self._ds = xr.open_dataset(file_path)

    @property
    def ds(self):
        return self._ds


class L1bImageBase(abc.ABC):
    """
    Interface from the L1b data to the retrieval
    """

    @abc.abstractmethod
    def sk2_geometries(self, alt_grid) -> (sk.Geometry1D, sk.ViewingGeometry):
        pass

    @abc.abstractmethod
    def skretrieval_l1(self):
        pass

    @abc.abstractproperty
    def lat(self):
        pass

    @abc.abstractproperty
    def lon(self):
        pass
