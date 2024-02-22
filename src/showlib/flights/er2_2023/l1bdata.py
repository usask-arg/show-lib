from __future__ import annotations

from pathlib import Path

import numpy as np
import sasktran2 as sk
import xarray as xr
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time
from scipy import interpolate
from skretrieval.core.radianceformat import RadianceGridded
from skretrieval.geodetic import geodetic
from skretrieval.util import rotation_matrix

from showlib.l1b.data import L1bImageBase


class L1bImage(L1bImageBase):
    def __init__(self, ds: xr.Dataset, fc: xr.Dataset, row_reduction_factor=1) -> None:
        self._apply_global_attributes()
        self._ds = ds.isel(los=slice(0, len(ds.los), row_reduction_factor))

        air_wavelength = 1e7 / self._ds.wavenumber
        vac_wavelenth = sk.optical.air_wavelength_to_vacuum_wavelength(air_wavelength)

        self._ds = self._ds.assign_coords({"wavenumber": 1e7 / vac_wavelenth})

        max_r = self._ds["radiance"].max(dim="wavenumber")

        noise = np.zeros_like(self._ds["radiance"])
        noise[np.newaxis, :] = max_r * 0.01
        self._ds["radiance_noise"] = (["wavenumber", "los"], noise)

        f = (
            fc["filter_correction"].to_numpy()
            * fc["pixel_response_correction"].to_numpy()
        )

        f = interpolate.interp1d(
            fc["wavenumber"].to_numpy(), f, kind="linear", fill_value="extrapolate"
        )(fc["wavenumber"].to_numpy() - 1.3)

        self._ds["radiance"] /= xr.DataArray(
            f,
            dims=["wavenumber"],
            coords={"wavenumber": self._ds["wavenumber"].to_numpy()},
        )
        self._ds["radiance_noise"] /= xr.DataArray(
            f,
            dims=["wavenumber"],
            coords={"wavenumber": self._ds["wavenumber"].to_numpy()},
        )

        self._obs_geo = geodetic()
        self._obs_geo.from_xyz(self._ds.observer_position[0, :])
        self._latitude = self._obs_geo.latitude
        self._longitude = self._obs_geo.longitude

    @property
    def lat(self):
        return self._latitude

    @property
    def lon(self):
        return self._longitude

    def _apply_global_attributes(self):
        pass

    @property
    def ds(self):
        return self._ds

    @classmethod
    def from_2024_balloon(cls, file_path: Path):
        """
        Loads in a single L1b file and provides access to the data

        """
        return cls(xr.open_dataset(file_path))

    def sk2_geometries(self, alt_grid) -> (sk.Geometry1D, sk.ViewingGeometry):
        # Calculate the viewing geometry
        viewing_geo = sk.ViewingGeometry()

        obs_pos = self._obs_geo.location

        time = Time(self._ds["utc"])
        sun = get_sun(time)

        geo = geodetic()
        alts = []
        angles = []
        rv = np.cross(
            self._ds.isel(los=0)["local_up"].values,
            self._ds.isel(los=0)["los_vectors"].values,
        )
        rv /= np.linalg.norm(rv)
        rm = rotation_matrix(rv, np.deg2rad(0.07))
        for i in range(len(self._ds.los)):
            geo.from_tangent_point(
                self._ds.observer_position.to_numpy()[i, :],
                rm @ self._ds.los_vectors.to_numpy()[i, :],
            )

            angles.append(
                np.rad2deg(
                    np.arccos(
                        np.dot(
                            self._ds.isel(los=i)["optical_axis"].values,
                            self._ds.isel(los=i)["los_vectors"].values,
                        )
                    )
                )
            )

            location = EarthLocation(lat=geo.latitude, lon=geo.longitude)
            cos_sza = np.cos(
                np.deg2rad(
                    90
                    - sun.transform_to(AltAz(obstime=time, location=location)).alt.value
                )
            )
            saa = (
                sun.transform_to(AltAz(obstime=time, location=location)).az.value - 180
            )

            viewing_geo.add_ray(
                sk.TangentAltitudeSolar(
                    geo.altitude, saa, self._obs_geo.altitude, cos_sza
                )
            )
            alts.append(geo.altitude)

        min_angle = np.argmin(angles)
        angles = np.array(angles)
        angles[:min_angle] *= -1
        self._ds["tangent_altitude"] = (["los"], alts)
        self._ds["angles"] = (["los"], angles)

        model_geometry = sk.Geometry1D(
            cos_sza=cos_sza,
            solar_azimuth=0,
            earth_radius_m=np.linalg.norm(obs_pos) - self._obs_geo.altitude,
            altitude_grid_m=alt_grid,
            interpolation_method=sk.InterpolationMethod.LinearInterpolation,
            geometry_type=sk.GeometryType.Spherical,
        )
        return model_geometry, viewing_geo

    def skretrieval_l1(self):
        return RadianceGridded(self._ds)
