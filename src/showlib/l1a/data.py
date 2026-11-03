from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from showlib.schemas import apply_metadata, check

SIMULATION_GROUP = "simulation"


def wrap_degrees(angle, lower: float) -> np.ndarray:
    """Wrap angles in degrees into [lower, lower + 360)."""
    return (np.asarray(angle, dtype=float) - lower) % 360.0 + lower


class L1AImage:
    """One SHOW detector image with its geolocation; see l1a.yaml."""

    def __init__(
        self,
        *,
        image: np.ndarray,
        time: datetime | np.datetime64,
        tangent_altitude: np.ndarray,
        tangent_latitude: np.ndarray,
        tangent_longitude: np.ndarray,
        spacecraft_latitude: float,
        spacecraft_longitude: float,
        spacecraft_altitude: float,
        solar_zenith_angle: np.ndarray,
        relative_solar_azimuth_angle: np.ndarray,
        los_azimuth_angle: np.ndarray,
        exposure_time: float,
        num_averaged_exposures: int = 1,
        saturated: np.ndarray | None = None,
    ):
        image_dims = ["los", "pixelcolumn"]
        data_vars = {
            "image": (image_dims, np.asarray(image, dtype=float)),
            "tangent_altitude": ("los", tangent_altitude),
            "tangent_latitude": ("los", tangent_latitude),
            "tangent_longitude": ("los", tangent_longitude),
            "spacecraft_latitude": ((), float(spacecraft_latitude)),
            "spacecraft_longitude": ((), float(spacecraft_longitude)),
            "spacecraft_altitude": ((), float(spacecraft_altitude)),
            "solar_zenith_angle": ("los", solar_zenith_angle),
            "relative_solar_azimuth_angle": (
                "los",
                wrap_degrees(relative_solar_azimuth_angle, -180.0),
            ),
            "los_azimuth_angle": ("los", wrap_degrees(los_azimuth_angle, 0.0)),
            "exposure_time": ((), float(exposure_time)),
            "num_averaged_exposures": ((), np.int64(num_averaged_exposures)),
        }
        if saturated is not None:
            data_vars["saturated"] = (image_dims, np.asarray(saturated, dtype=bool))
        self._ds = xr.Dataset(data_vars, coords={"time": np.datetime64(time, "ns")})

    @property
    def ds(self) -> xr.Dataset:
        return self._ds


class L1ADataSet:
    """An L1A granule: consecutive images along time, plus optional simulation data."""

    def __init__(self, ds: xr.Dataset, simulation: xr.Dataset | None = None):
        check(_as_tree(ds, simulation), "l1a")
        self._ds = ds
        self._simulation = simulation

    @classmethod
    def from_images(
        cls, images: list[L1AImage], simulation: xr.Dataset | None = None
    ) -> L1ADataSet:
        ds = xr.concat(
            [image.ds for image in images],
            dim="time",
            data_vars="all",
            coords="minimal",
            join="exact",
        )
        tree = apply_metadata(_as_tree(ds, simulation), "l1a")
        return cls(*_from_tree(tree))

    @classmethod
    def from_file(cls, path: Path) -> L1ADataSet:
        with xr.open_datatree(path) as tree:
            return cls(*_from_tree(tree.load()))

    @property
    def ds(self) -> xr.Dataset:
        return self._ds

    @property
    def simulation(self) -> xr.Dataset | None:
        return self._simulation

    def save(self, path: Path) -> None:
        _as_tree(self._ds, self._simulation).to_netcdf(path)


def _as_tree(ds: xr.Dataset, simulation: xr.Dataset | None) -> xr.DataTree:
    nodes = {"/": ds}
    if simulation is not None:
        nodes[f"/{SIMULATION_GROUP}"] = simulation
    return xr.DataTree.from_dict(nodes)


def _from_tree(tree: xr.DataTree) -> tuple[xr.Dataset, xr.Dataset | None]:
    simulation = tree.children.get(SIMULATION_GROUP)
    return (
        tree.to_dataset(inherit=False),
        None if simulation is None else simulation.to_dataset(inherit=False),
    )
