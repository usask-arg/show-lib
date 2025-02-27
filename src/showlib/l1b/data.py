from __future__ import annotations

import abc
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sasktran2 as sk
import xarray as xr
from sasktran2.geodetic import WGS84
from sasktran2.viewinggeo.ecef import ecef_to_sasktran2_ray
from skretrieval.core.radianceformat import RadianceGridded
from skretrieval.retrieval.observation import Observation
from skretrieval.util import rotation_matrix


class L1bImage(Observation):
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
        los_azimuth_angle: np.array,
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
        ds["los_azimuth_angle"] = xr.DataArray(los_azimuth_angle, dims=["los"])
        return cls(ds)

    def __init__(
        self,
        ds: xr.Dataset,
        low_alt=0,
        high_alt=100000,
        low_wvnumber_filter=0,
        high_wvnumber_filter=1e10,
        override_los=False,
    ):
        self._ds = ds
        self._low_alt = low_alt
        self._high_alt = high_alt
        self._low_wvnumber_filter = low_wvnumber_filter
        self._high_wvnumber_filter = high_wvnumber_filter
        self._override_los = override_los

    @property
    def ds(self):
        return self._ds

    def sk2_geometry(self) -> dict[sk.ViewingGeometry]:
        viewing_geo = sk.ViewingGeometry()

        good_alt = (self._ds.tangent_altitude.to_numpy() > self._low_alt) & (
            self._ds.tangent_altitude.to_numpy() < self._high_alt
        )

        for i in range(len(self._ds.los[good_alt])):
            if self._override_los:
                obs_geo = WGS84()
                obs_geo.from_lat_lon_alt(
                    self._ds["spacecraft_latitude"],
                    self._ds["spacecraft_longitude"],
                    self._ds["spacecraft_altitude"],
                )

                tangent_geo = WGS84()
                tangent_geo.from_lat_lon_alt(
                    self._ds["tangent_latitude"].to_numpy()[good_alt][i],
                    self._ds["tangent_longitude"].to_numpy()[good_alt][i],
                    self._ds["tangent_altitude"].to_numpy()[good_alt][i],
                )

                look_v = tangent_geo.location - obs_geo.location
                look_v = look_v / np.linalg.norm(look_v)

                rv = np.cross(
                    obs_geo.local_up,
                    look_v,
                )
                rv /= np.linalg.norm(rv)
                rm = rotation_matrix(rv, np.deg2rad(0.21))

                look_v = rm @ look_v

                viewing_geo.add_ray(
                    ecef_to_sasktran2_ray(
                        obs_geo.location,
                        look_v,
                        pd.to_datetime(self._ds["time"].to_numpy()),
                        solar_handler=sk.solar.SolarGeometryHandlerAstropy(),
                    )
                )
            else:
                viewing_geo.add_ray(
                    sk.TangentAltitudeSolar(
                        self._ds["tangent_altitude"].to_numpy()[good_alt][i],
                        np.deg2rad(
                            self._ds["relative_solar_azimuth_angle"].to_numpy()[
                                good_alt
                            ][i]
                        ),
                        float(self._ds["spacecraft_altitude"]),
                        np.cos(
                            np.deg2rad(
                                self._ds["solar_zenith_angle"].to_numpy()[good_alt][i]
                            )
                        ),
                    )
                )

        return {"meas": viewing_geo}

    def skretrieval_l1(self, *args, **kwargs):
        ds = xr.Dataset()

        wvnum = (self._ds["left_wavenumber"].to_numpy()[0]) + np.arange(
            0, len(self._ds.sample)
        ) * self._ds["wavenumber_spacing"].to_numpy()[0]

        good = (wvnum > self._low_wvnumber_filter) & (
            wvnum < self._high_wvnumber_filter
        )

        good_alt = (self._ds.tangent_altitude.to_numpy() > self._low_alt) & (
            self._ds.tangent_altitude.to_numpy() < self._high_alt
        )

        if self._ds["radiance"].mean() > 1e9:
            logging.warning(
                "Radiance appears to not be in units of W/m^2/nm/sr, trying to convert."
            )
            c = 299792458  # m/s, speed of light
            h = 6.62607015e-34  # J/Hz, planck's constant
            photon_energy = c * h * wvnum[good] / (1e-2)  # number of J per photon
            rad_conversion_factors = (
                np.array(photon_energy) * 100 * 100
            )  # multiply osiris data by unit factor to obtain units of # W/m^2/nm/ster

            wvlen = 1e7 / wvnum[good]
            # Convert from /cm-1 to /nm
            rad_conversion_factors = rad_conversion_factors * (1e7 / wvlen**2)

        else:
            rad_conversion_factors = np.ones(len(wvnum[good]))

        ds["radiance"] = xr.DataArray(
            self._ds["radiance"].to_numpy()[np.ix_(good, good_alt)]
            * rad_conversion_factors[:, np.newaxis],
            dims=["wavenumber", "los"],
        )
        ds["radiance_noise"] = xr.DataArray(
            self._ds["radiance_noise"].to_numpy()[np.ix_(good, good_alt)]
            * rad_conversion_factors[:, np.newaxis],
            dims=["wavenumber", "los"],
        )

        ds["tangent_altitude"] = self._ds["tangent_altitude"][good_alt]
        ds.coords["wavenumber"] = wvnum[good]

        return {"meas": RadianceGridded(ds)}

    @abc.abstractmethod
    def sample_wavelengths(self) -> dict[np.array]:
        """
        The sample wavelengths for the observation in [nm]

        Returns
        -------
        dict[np.array]
        """
        l1 = self.skretrieval_l1()

        return {"meas": 1e7 / l1["meas"].data.wavenumber}

    @abc.abstractmethod
    def reference_cos_sza(self) -> dict[float]:
        """
        The reference cosine of the solar zenith angle for the observation

        Returns
        -------
        dict[float]
        """
        return {"meas": np.cos(self._ds["solar_zenith_angle"].mean())}

    @abc.abstractmethod
    def reference_latitude(self) -> dict[float]:
        """
        The reference latitude for the observation

        Returns
        -------
        dict[float]
        """
        return {"meas": float(self._ds["tangent_latitude"].mean())}

    @abc.abstractmethod
    def reference_longitude(self) -> dict[float]:
        """
        The reference longitude for the observation

        Returns
        -------
        dict[float]
        """
        return {"meas": float(self._ds["tangent_longitude"].mean())}

    def append_information_to_l1(self, l1: dict[RadianceGridded], **kwargs) -> None:
        """
        A method that allows for the observation to append information to the L1 data
        simulated by the forward model. Useful for adding things that are in the real L1 data
        to the simulations that may be useful inside the measurement vector.

        Parameters
        ----------
        l1 : dict[RadianceGridded]
        """
        good_alt = (self._ds.tangent_altitude.to_numpy() > self._low_alt) & (
            self._ds.tangent_altitude.to_numpy() < self._high_alt
        )
        l1["meas"].data["tangent_altitude"] = (
            ["los"],
            self._ds["tangent_altitude"].to_numpy()[good_alt],
        )


class L1bFileWriter:
    def __init__(self, l1b_data: list[L1bImage]) -> None:
        self._data = xr.concat([l1b._ds for l1b in l1b_data], dim="time")

    def _apply_global_attributes(self):
        pass

    def save(self, out_file: Path):
        self._data.to_netcdf(out_file.as_posix())


class L1bDataSet:
    def __init__(self, ds: xr.Dataset, name: str, parent: Path):
        """
        Loads in a single L1b file and provides access to the data

        """
        self._ds = ds
        self._name = name
        self._parent = parent

    @classmethod
    def from_image(cls, image: L1bImage):
        return cls(xr.concat([l1b._ds for l1b in [image]], dim="time"), "", None)

    @classmethod
    def from_file(cls, file_path: Path):
        return cls(xr.open_dataset(file_path), file_path.stem, file_path.parent.parent)

    @property
    def ds(self):
        return self._ds

    def image(
        self, sample: int, row_reduction: int = 1, low_alt=0, high_alt=100000, **kwargs
    ):
        return L1bImage(
            self._ds.isel(time=sample).isel(los=slice(None, None, row_reduction)),
            low_alt=low_alt,
            high_alt=high_alt,
            **kwargs,
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
