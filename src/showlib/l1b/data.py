from __future__ import annotations

import abc
from datetime import datetime
from pathlib import Path

import numpy as np
import sasktran2 as sk
import xarray as xr
from skretrieval.core.radianceformat import RadianceGridded
from skretrieval.geodetic import geodetic


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


class L1bImage(L1bImageBase):
    @classmethod
    def from_np_arrays(
        cls,
        radiance: np.array,
        radiance_noise: np.array,
        tangent_altitude: np.array,
        tangent_latitude: np.array,
        tangent_longitude: np.array,
        left_wavenumber: np.array,
        wavenumber_spacing: np.array,
        time: datetime,
        observer_latitude: float,
        observer_longitude: float,
        observer_altitude: float,
        sza: np.array,
        saa: np.array,
    ):
        ds = xr.Dataset()

        ds["radiance"] = xr.DataArray(radiance, dims=["sample", "los"])
        ds["radiance_noise"] = xr.DataArray(radiance_noise, dims=["sample", "los"])
        ds["tangent_altitude"] = xr.DataArray(tangent_altitude, dims=["los"])
        ds["tangent_latitude"] = xr.DataArray(tangent_latitude, dims=["los"])
        ds["tangent_longitude"] = xr.DataArray(tangent_longitude, dims=["los"])

        ds["time"] = time

        ds["left_wavenumber"] = xr.DataArray(left_wavenumber, dims=["los"])
        ds["wavenumber_spacing"] = xr.DataArray(wavenumber_spacing, dims=["los"])

        ds["spacecraft_latitude"] = observer_latitude
        ds["spacecraft_longitude"] = observer_longitude
        ds["spacecraft_altitude"] = observer_altitude

        ds["solar_zenith_angle"] = xr.DataArray(sza, dims=["los"])
        ds["relative_solar_azimuth_angle"] = xr.DataArray(saa, dims=["los"])

        return cls(ds)

    def __init__(self, ds: xr.Dataset, low_alt=0, high_alt=100000):
        self._ds = ds
        self._low_alt = low_alt
        self._high_alt = high_alt

    @property
    def ds(self):
        return self._ds

    def sk2_geometries(self, alt_grid) -> (sk.Geometry1D, sk.ViewingGeometry):
        geo = geodetic()

        geo.from_lat_lon_alt(self.lat, self.lon, 0)
        earth_radius = np.linalg.norm(geo.location)

        cos_sza = np.cos(self._ds["solar_zenith_angle"].mean())

        model_geometry = sk.Geometry1D(
            cos_sza=cos_sza,
            solar_azimuth=0,
            earth_radius_m=earth_radius,
            altitude_grid_m=alt_grid,
            interpolation_method=sk.InterpolationMethod.LinearInterpolation,
            geometry_type=sk.GeometryType.Spherical,
        )

        viewing_geo = sk.ViewingGeometry()

        good_alt = (self._ds.tangent_altitude.to_numpy() > self._low_alt) & (
            self._ds.tangent_altitude.to_numpy() < self._high_alt
        )

        for i in range(len(self._ds.los[good_alt])):
            viewing_geo.add_ray(
                sk.TangentAltitudeSolar(
                    self._ds["tangent_altitude"].to_numpy()[good_alt][i],
                    np.deg2rad(
                        self._ds["relative_solar_azimuth_angle"].to_numpy()[good_alt][i]
                    ),
                    float(self._ds["spacecraft_altitude"]),
                    np.cos(
                        np.deg2rad(
                            self._ds["solar_zenith_angle"].to_numpy()[good_alt][i]
                        )
                    ),
                )
            )

        return model_geometry, viewing_geo

    def skretrieval_l1(self):
        ds = xr.Dataset()

        wvnum = (
            self._ds["left_wavenumber"].to_numpy()[0]
            + np.arange(0, len(self._ds.sample))
            * self._ds["wavenumber_spacing"].to_numpy()[0]
        )

        good = (wvnum > 7310) & (wvnum < 7330)

        good_alt = (self._ds.tangent_altitude.to_numpy() > self._low_alt) & (
            self._ds.tangent_altitude.to_numpy() < self._high_alt
        )

        ds["radiance"] = xr.DataArray(
            self._ds["radiance"].to_numpy()[np.ix_(good, good_alt)],
            dims=["wavenumber", "los"],
        )
        ds["radiance_noise"] = xr.DataArray(
            self._ds["radiance_noise"].to_numpy()[np.ix_(good, good_alt)],
            dims=["wavenumber", "los"],
        )

        ds["tangent_altitude"] = self._ds["tangent_altitude"][good_alt]
        ds.coords["wavenumber"] = wvnum[good]

        return RadianceGridded(ds)

    @property
    def lat(self):
        return float(self._ds["tangent_latitude"].mean())

    @property
    def lon(self):
        return float(self._ds["tangent_longitude"].mean())


class L1bFileWriter:
    def __init__(self, l1b_data: list[L1bImage]) -> None:
        self._data = xr.concat([l1b._ds for l1b in l1b_data], dim="time")

    def _apply_global_attributes(self):
        pass

    def save(self, out_file: Path):
        self._data.to_netcdf(out_file.as_posix())


class L1bDataSet:
    def __init__(self, file_path: Path):
        """
        Loads in a single L1b file and provides access to the data

        """
        self._ds = xr.open_dataset(file_path)
        self._name = file_path.stem
        self._parent = file_path.parent.parent

    @property
    def ds(self):
        return self._ds

    def image(self, sample: int, row_reduction: int = 1, low_alt=0, high_alt=100000):
        return L1bImage(
            self._ds.isel(time=sample).isel(los=slice(None, None, row_reduction)),
            low_alt=low_alt,
            high_alt=high_alt,
        )

    @property
    def l2_por_path(self):
        return self._parent.joinpath("por").joinpath(
            self._name.replace("L1B", "L2_POR").replace("Radiances", "Wvapor") + ".nc"
        )

    @property
    def l2_path(self):
        return self._parent.joinpath("l2").joinpath(
            self._name.replace("L1B", "L2").replace("Radiances", "Wvapor") + ".nc"
        )
