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
from skretrieval.time.mjd import mjd_to_datetime
from skretrieval.util import rotation_matrix

from showlib.l1b.data import L1bFileWriter, L1bImage


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
            writer = L1bFileWriter(l1bs)
            writer.save(
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
        lwv = selected["wavenumber"].to_numpy()[0]
        spacing = float(selected["wavenumber"].diff(dim="wavenumber").mean())
        l1b = L1bImage.from_np_arrays(
            selected["radiance"].to_numpy(),
            selected["radiance_noise"].to_numpy(),
            np.array(alts),
            np.array(lats),
            np.array(lons),
            np.ones_like(alts) * lwv,
            np.ones_like(alts) * spacing,
            time,
            geo.latitude,
            geo.longitude,
            geo.altitude,
            np.array(szas),
            np.array(saas),
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
