from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import sasktran2 as sk
import xarray as xr
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time
from scipy import interpolate
from skretrieval.geodetic import geodetic
from skretrieval.util import rotation_matrix

from showlib.l1b.data import L1bDataSet, L1bImage
from showlib.timeutil import mjd_to_datetime

_PLANCK_CONSTANT_J_S = 6.62607015e-34
_SPEED_OF_LIGHT_M_S = 299792458.0


def _unit_conversion(radiance: np.ndarray, wavenumber: np.ndarray) -> np.ndarray:
    """Factor converting the ER-2 radiance to W m-2 sr-1 nm-1, per wavenumber.

    The ER-2 radiances may be in photons s-1 cm-2 sr-1 (cm-1)-1; L1B is always
    W m-2 sr-1 nm-1. A mean above 1e9 is treated as photon units, as L1bImage did
    before the L1B schema fixed the units.
    """
    if np.nanmean(radiance) <= 1e9:
        return np.ones_like(wavenumber)
    photon_energy_j = _PLANCK_CONSTANT_J_S * _SPEED_OF_LIGHT_M_S * wavenumber * 100
    per_nm = 1e7 / (1e7 / wavenumber) ** 2
    return photon_energy_j * 1e4 * per_nm


def convert_full_l1b_to_sds(
    file: Path, out_folder: Path, fc: xr.Dataset, granularity_minutes=5
):
    ds = xr.open_dataset(file)

    # First convert air wavenumber to vacuum
    if "wavelength" in ds.dims:
        air_wavelength = ds.wavelength
        vac_wavelenth = sk.optical.air_wavelength_to_vacuum_wavelength(air_wavelength)

        ds = (
            ds.assign_coords({"wavelength": 1e7 / vac_wavelenth})
            .rename_dims({"wavelength": "wavenumber"})
            .rename_vars({"wavelength": "wavenumber"})
            .isel(wavenumber=slice(None, None, -1))
        )
    else:
        air_wavelength = 1e7 / ds.wavenumber
        vac_wavelenth = sk.optical.air_wavelength_to_vacuum_wavelength(air_wavelength)

        ds = ds.assign_coords({"wavenumber": 1e7 / vac_wavelenth})

    f = fc["filter_correction"].to_numpy() * fc["pixel_response_correction"].to_numpy()

    ds["radiance"] = np.sqrt(ds["radiance"] ** 2 + ds["imaginary"] ** 2)

    f = interpolate.interp1d(
        fc["wavenumber"].to_numpy(), f, kind="linear", fill_value="extrapolate"
    )(fc["wavenumber"].to_numpy() - 1.3)

    ds["radiance_noise"] = ds["radiance"] / ds["SNR"]

    ds["radiance"] /= xr.DataArray(
        f,
        dims=["wavenumber"],
        coords={"wavenumber": ds["wavenumber"].to_numpy()},
    )
    ds["radiance_noise"] /= xr.DataArray(
        f,
        dims=["wavenumber"],
        coords={"wavenumber": ds["wavenumber"].to_numpy()},
    )

    times = ds["mjd"].to_numpy()[:, 0]

    start_time = mjd_to_datetime(times[0])
    granularity_delta = timedelta(minutes=granularity_minutes)

    l1bs = []
    for i in range(0, len(times), 5):
        time = mjd_to_datetime(times[i])
        if time - start_time > granularity_delta:
            L1bDataSet.from_images(l1bs).save(
                out_folder.joinpath(
                    f"HAWC_H2OL_Radiances_L1B_{start_time.strftime('%Y-%m-%dT%H:%M:%SZ')}.v0_0_1.STD.nc"
                )
            )

            l1bs = []
            start_time = time

        selected = ds.isel(sample=i).isel(los=slice(None, None, 5))

        obs_pos = selected.observer_position.to_numpy()[0, :]

        sun = get_sun(Time(time))

        geo = geodetic()
        alts = []
        saas = []
        szas = []
        lons = []
        lats = []
        rv = np.cross(
            selected.isel(los=0)["local_up"].values,
            selected.isel(los=0)["los_vectors"].values,
        )
        rv /= np.linalg.norm(rv)
        rm = rotation_matrix(rv, np.deg2rad(0.07))
        for j in range(len(selected.los)):
            geo.from_tangent_point(
                selected.observer_position.to_numpy()[j, :],
                rm @ selected.los_vectors.to_numpy()[j, :],
            )

            location = EarthLocation(lat=geo.latitude, lon=geo.longitude)
            sza = (
                90 - sun.transform_to(AltAz(obstime=time, location=location)).alt.value
            )
            saa = (
                sun.transform_to(AltAz(obstime=time, location=location)).az.value - 180
            )

            szas.append(sza)
            saas.append(saa)
            lons.append(geo.longitude)
            lats.append(geo.latitude)

            alts.append(geo.altitude)

        geo.from_xyz(obs_pos)
        wavenumber = selected["wavenumber"].to_numpy()
        radiance = selected["radiance"].transpose("los", "wavenumber").to_numpy()
        noise = selected["radiance_noise"].transpose("los", "wavenumber").to_numpy()
        conversion = _unit_conversion(radiance, wavenumber)
        l1b = L1bImage.from_np_arrays(
            radiance=radiance * conversion,
            radiance_noise=noise * conversion,
            wavenumber=wavenumber,
            tangent_altitude=np.array(alts),
            tangent_latitude=np.array(lats),
            tangent_longitude=np.array(lons),
            time=time,
            spacecraft_latitude=geo.latitude,
            spacecraft_longitude=geo.longitude,
            spacecraft_altitude=geo.altitude,
            solar_zenith_angle=np.array(szas),
            relative_solar_azimuth_angle=np.array(saas),
            # Not computed by this converter
            los_azimuth_angle=np.full(len(alts), np.nan),
            # The ER-2 apodization and noise model are not recorded in the source
            # files; the L1B schema requires them, so saving fails until they are set.
            processing={},
        )
        l1bs.append(l1b)


if __name__ == "__main__":
    convert_full_l1b_to_sds(
        Path("/Users/dannyz/Downloads/ScienceFlight1SanDiego_l1b_cm_1_v0.nc"),
        Path(
            "/Users/dannyz/OneDrive - University of Saskatchewan/SHOW/er2_2023/sci_flight/l1b_mag"
        ),
        xr.open_dataset(
            "/Users/dannyz/Downloads/show_example_for_zawada_one_sample.nc"
        ),
        granularity_minutes=1,
    )
