from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import interpolate
from skretrieval.util import linear_interpolating_matrix


def get_tropopause_height(alt, temperature):
    # Determines the thermal tropopause from temperature profiles in the input dataset.
    # Converted from matlab to python and adapted for this script from:
    # https://arggit.usask.ca/SVN/Repos_OSIRISOps/-/blob/master/matlab_OsirisOperations/ncep_tropopauseheight/GetTropopauseHeight.m

    dT_lim = -2  # lapse rate limit
    num_levels = len(alt)

    # height derivative indexing arrays: use Forward Difference at top/bottom and Central difference otherwise
    up = np.zeros(num_levels, dtype="int")
    down = np.zeros(num_levels, dtype="int")
    up[0:-1] = np.arange(1, num_levels)
    up[-1] = num_levels - 1
    down[1::] = np.arange(0, num_levels - 1)
    down[0] = 0

    dz = alt[up] - alt[down]  # height derivative (km)

    dTdz = (temperature[up] - temperature[down]) / dz
    return fit_tropopause(dTdz, alt, dT_lim)


def fit_tropopause(param, alt, threshold):
    # base on FitTropopause2 from
    # https://arggit.usask.ca/SVN/Repos_OSIRISOps/-/blob/master/matlab_OsirisOperations/ncep_tropopauseheight/GetTropopauseHeight.m

    # find the first point greater than 5 km (try to skip over low altitude temperature inversion issues)
    idx = 0
    while alt[idx] < 5:
        idx += 1

    # find the points below 26 km
    last_idx = idx
    while alt[last_idx] < 26:
        last_idx += 1

    while (
        idx >= 0 and param[idx] > threshold
    ):  # if we actually above the tropopause, it occasionally happens
        idx -= 1  # then step down in altitude until we are below it

    if idx >= 0:  # if we found at least 1 point "below the tropopause" then we are good
        data_index = slice(idx, last_idx + 1)  # get the points to fit
        the_alts = alt[data_index]  # get the altitudes
        the_param = param[data_index]  # get the parameter

        # get the first point above the tropopause
        trop_idx = 0
        while the_param[trop_idx] <= threshold:
            trop_idx += 1

        x1 = the_alts[trop_idx - 1]
        x2 = the_alts[trop_idx]
        y1 = the_param[trop_idx - 1]
        y2 = the_param[trop_idx]
        trop_alt = x1 + (threshold - y1) * ((x2 - x1) / (y2 - y1))
        if trop_alt < x1 or trop_alt > x2:  # make sure we have linearly interpolated
            trop_alt = np.nan  # this should never happen unless there is a logic error

    else:
        trop_alt = np.nan  # this should never happen in Earths atmosphere,

    return trop_alt


class MERRA2:
    def __init__(self, data_folder: Path):
        self._data_folder = data_folder
        self._time = None

    def _load(self, time: datetime, latitude, longitude):
        if (self._time is None) or (
            (self._time.year != time.year)
            or (self._time.month != time.month)
            or (self._time.day != time.day)
        ):
            year = time.year
            month = time.month
            day = time.day

            self._time = time

            datestr = f"{year}{str(month).zfill(2)}{str(day).zfill(2)}"

            possible_files = list(
                self._data_folder.glob(f"MERRA2_*.inst3_3d_asm_Nv.{datestr}.nc4")
            )

            if len(possible_files) == 0:
                msg = f"Could not find Merra file for date string {datestr}"
                raise OSError(msg)

            self._tds = xr.open_dataset(possible_files[0].as_posix()).interp(time=time)

        self._ds = self._tds.interp(lat=latitude, lon=longitude)

    def calculate_parameters(
        self, time: datetime, latitude: float, longitude: float, altitude_grid: np.array
    ):
        self._load(time, latitude, longitude)
        profile_data = self._ds

        # Start by calculating the dry air number density, we want to ensure that the vertical integral of
        # number density on the altitude grid matches what would be expected from the pressure levels
        pressure_level_boundaries = np.cumsum(
            np.concatenate(([0.01], profile_data["DELP"].to_numpy()[::-1]))
        )
        pressure_level_boundaries[-1] = profile_data["PS"].to_numpy()

        pressure_mids = profile_data["PL"].to_numpy()

        alt_mids = profile_data["H"].to_numpy()

        # Interpolate quantities to the boundaries of layers
        alt_boundaries = interpolate.interp1d(
            np.log(pressure_mids), alt_mids, fill_value="extrapolate"
        )(np.log(pressure_level_boundaries))

        s_idx = np.argsort(np.concatenate((alt_mids, alt_boundaries)))
        full_altitude_grid = np.concatenate((alt_mids, alt_boundaries))[s_idx]
        full_pressure_grid = np.concatenate((pressure_mids, pressure_level_boundaries))[
            s_idx
        ]

        profile_data = profile_data.swap_dims({"lev": "H"}).interp(
            H=full_altitude_grid, kwargs={"fill_value": "extrapolate"}
        )
        profile_data["PL"].values = full_pressure_grid

        air_dens = (
            profile_data["PL"]
            / (8.31446261815324 * profile_data["T"])
            * 6.0221408e23
            / 1e6
            * (1 - profile_data["QV"])
        )

        water_dens = air_dens * profile_data["QV"] / (1 - profile_data["QV"])

        water_vmr = water_dens / air_dens

        # Create a new output altitude grid spanned by the input boundaries
        new_alt_grid = altitude_grid[
            (altitude_grid > full_altitude_grid[0])
            & (altitude_grid < full_altitude_grid[-1])
        ]
        new_alt_grid = np.unique(
            np.sort(
                np.concatenate(
                    ([full_altitude_grid[0]], new_alt_grid, [full_altitude_grid[-1]])
                )
            )
        )

        # TODO: Figure out how to use this with uneven grids
        # reduction_matrix = linear_reduction_matrix(new_alt_grid.astype('float'), profile_data['H'].values)

        reduction_matrix = linear_interpolating_matrix(
            profile_data["H"].to_numpy(), new_alt_grid.astype("float")
        )

        reduced_pressure = np.exp(reduction_matrix @ np.log(profile_data["PL"].values))

        return xr.Dataset(
            {
                "air_density": (["altitude"], reduction_matrix @ air_dens.to_numpy()),
                "pressure": (["altitude"], reduced_pressure),
                "temperature": (
                    ["altitude"],
                    reduction_matrix @ profile_data["T"].to_numpy(),
                ),
                "water_vmr": (["altitude"], reduction_matrix @ water_vmr.to_numpy()),
                "tropopause_altitude": get_tropopause_height(
                    profile_data["H"].to_numpy() / 1000, profile_data["T"].to_numpy()
                )
                * 1000,
            },
            coords={"altitude": new_alt_grid},
        )


if __name__ == "__main__":
    test_time = datetime(year=2010, month=4, day=9)

    data = MERRA2(
        test_time,
        Path(r"/Users/dannyz/OneDrive - University of Saskatchewan/IFTS/data"),
    )

    anc = data.calculate_parameters(10, 30, np.arange(0, 100001, 2000))
