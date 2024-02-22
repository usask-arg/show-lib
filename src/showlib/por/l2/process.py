from __future__ import annotations

from pathlib import Path

import numpy as np
import sasktran as sk
import xarray as xr
from skretrieval.time.mjd import datetime64_to_mjd

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


if __name__ == "__main__":
    in_folder = Path(
        "/Users/dannyz/OneDrive - University of Saskatchewan/SHOW/er2_2023/sci_flight/l1b/"
    )

    for file in in_folder.iterdir():
        if file.suffix == ".nc":
            l2_por(
                file,
                Path(
                    "/Users/dannyz/OneDrive - University of Saskatchewan/SHOW/er2_2023/sci_flight/por"
                ),
            )
