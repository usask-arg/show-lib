from __future__ import annotations

from pathlib import Path

import numpy as np
import sasktran as sk
import xarray as xr
from skretrieval.time.datetime64 import datetime64_to_datetime
from skretrieval.time.mjd import datetime64_to_mjd

from showlib.por.l2.merra import MERRA2

ALT_GRID = np.arange(0, 100001, 250)


def l2_por(l1b_granule: Path, out_folder: Path):
    """
    Process a single L2 granule and save the results to the out_folder

    Parameters
    ----------
    l2_granule: Path
        Path to the L2 granule to process

    out_folder: Path
        Folder to save the results to
    """
    ds = xr.open_dataset(l1b_granule)
    out_file = out_folder.joinpath(
        l1b_granule.stem.replace("L1B", "L2_POR").replace("Radiances", "Wvapor") + ".nc"
    )

    pressure = np.zeros((len(ds.time), len(ALT_GRID)))
    temperature = np.zeros((len(ds.time), len(ALT_GRID)))

    if not out_file.exists():
        msis = sk.MSIS90()

        for sample in range(len(ds.time)):
            selected = ds.isel(time=sample)

            lat = float(selected["tangent_latitude"].mean())
            lon = float(selected["tangent_longitude"].mean())

            mjd = datetime64_to_mjd(selected["time"].to_numpy())

            pressure[sample, :] = msis.get_parameter(
                "SKCLIMATOLOGY_PRESSURE_PA", lat, lon, ALT_GRID, mjd
            )
            temperature[sample, :] = msis.get_parameter(
                "SKCLIMATOLOGY_TEMPERATURE_K", lat, lon, ALT_GRID, mjd
            )

        out = xr.Dataset()

        out["temperature"] = xr.DataArray(temperature, dims=["time", "altitude"])
        out["pressure"] = xr.DataArray(pressure, dims=["time", "altitude"])

        out.coords["time"] = ds.time
        out.coords["altitude"] = ALT_GRID

        out.to_netcdf(out_file.as_posix())


def l2_por_merra(l1b_granule: Path, out_folder: Path):
    """
    Process a single L2 granule and save the results to the out_folder

    Parameters
    ----------
    l2_granule: Path
        Path to the L2 granule to process

    out_folder: Path
        Folder to save the results to
    """
    merra2 = MERRA2(Path("/home/dannyz/mnts/utls3/MERRA2"))

    ds = xr.open_dataset(l1b_granule)
    out_file = out_folder.joinpath(
        l1b_granule.stem.replace("L1B", "L2_POR").replace("Radiances", "Wvapor") + ".nc"
    )

    pressure = np.zeros((len(ds.time), len(ALT_GRID)))
    temperature = np.zeros((len(ds.time), len(ALT_GRID)))
    vmr = np.zeros((len(ds.time), len(ALT_GRID)))
    tropopause_alt = np.zeros(len(ds.time))

    if not out_file.exists():
        for sample in range(len(ds.time)):
            selected = ds.isel(time=sample)

            lat_min = float(selected["tangent_latitude"].min())
            lat_max = float(selected["tangent_latitude"].max())

            lon = (float(selected["tangent_longitude"].mean()) + 180) % 360 - 180

            merra_min = merra2.calculate_parameters(
                datetime64_to_datetime(selected["time"].to_numpy()),
                lat_min,
                lon,
                ALT_GRID,
            ).interp(altitude=ALT_GRID)
            merra_max = merra2.calculate_parameters(
                datetime64_to_datetime(selected["time"].to_numpy()),
                lat_max,
                lon,
                ALT_GRID,
            ).interp(altitude=ALT_GRID)

            grid_lats = np.interp(
                ALT_GRID,
                selected.tangent_altitude.to_numpy(),
                selected.tangent_latitude.to_numpy(),
                left=lat_min,
                right=lat_max,
            )

            w_low = 1 - (grid_lats - lat_min) / (lat_max - lat_min)
            w_high = 1 - w_low

            weighted_ds = (
                xr.DataArray(w_low, dims=["altitude"]) * merra_min
                + xr.DataArray(w_high, dims=["altitude"]) * merra_max
            )

            pressure[sample, :] = weighted_ds["pressure"].to_numpy()
            temperature[sample, :] = weighted_ds["temperature"].to_numpy()
            vmr[sample, :] = weighted_ds["water_vmr"].to_numpy()

            tropopause_alt[sample] = float(merra_min["tropopause_altitude"])

            out = xr.Dataset()

        out["temperature"] = xr.DataArray(temperature, dims=["time", "altitude"])
        out["pressure"] = xr.DataArray(pressure, dims=["time", "altitude"])
        out["h2o_vmr"] = xr.DataArray(vmr, dims=["time", "altitude"])
        out["tropopause_altitude"] = xr.DataArray(tropopause_alt, dims=["time"])

        out.coords["time"] = ds.time
        out.coords["altitude"] = ALT_GRID

        out.to_netcdf(out_file.as_posix())


if __name__ == "__main__":
    in_folder = Path(
        "/datastore/root/research_projects/SHOW/er2_2023/data/science_flight_1_testing/l1b/"
    )

    for file in in_folder.glob("HAWC_H2OL*"):
        if file.suffix == ".nc":
            l2_por_merra(
                file,
                Path(
                    "/datastore/root/research_projects/SHOW/er2_2023/data/science_flight_1_testing/por/"
                ),
            )
