"""SHOW L2 profiles and granules; the format is src/showlib/schemas/l2.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from showlib.schemas import apply_metadata, check


@dataclass(frozen=True)
class AerosolProfile:
    """The co-retrieved aerosol extinction of one profile, on its altitude grid."""

    extinction: np.ndarray
    prior: np.ndarray
    noise_error: np.ndarray
    solution_error: np.ndarray
    averaging_kernel: np.ndarray
    retrieval_space: str
    wavelength_nm: float


class L2Profile:
    """One retrieved profile, the L2 granule at one time; see l2.yaml."""

    def __init__(
        self,
        *,
        time: datetime | np.datetime64,
        latitude: float,
        longitude: float,
        altitude: np.ndarray,
        h2o_vmr: np.ndarray,
        h2o_vmr_prior: np.ndarray,
        h2o_vmr_noise_error: np.ndarray,
        h2o_vmr_solution_error: np.ndarray,
        h2o_vmr_averaging_kernel: np.ndarray,
        h2o_retrieval_space: str,
        tropopause_altitude: float,
        lower_altitude_limit: float,
        upper_altitude_limit: float,
        num_los: int,
        cost: float,
        optimizer_status: int,
        num_function_evaluations: int,
        aerosol: AerosolProfile | None = None,
    ):
        altitude = np.asarray(altitude, dtype=float)
        profile_dims = ["altitude"]
        kernel_dims = ["altitude", "altitude2"]
        data_vars = {
            "latitude": ((), float(latitude)),
            "longitude": ((), float(longitude)),
            "h2o_vmr": (profile_dims, h2o_vmr),
            "h2o_vmr_prior": (profile_dims, h2o_vmr_prior),
            "h2o_vmr_noise_error": (profile_dims, h2o_vmr_noise_error),
            "h2o_vmr_solution_error": (profile_dims, h2o_vmr_solution_error),
            "h2o_vmr_averaging_kernel": (kernel_dims, h2o_vmr_averaging_kernel),
            "tropopause_altitude": ((), float(tropopause_altitude)),
            "lower_altitude_limit": ((), float(lower_altitude_limit)),
            "upper_altitude_limit": ((), float(upper_altitude_limit)),
            "num_los": ((), np.int64(num_los)),
            "cost": ((), float(cost)),
            "optimizer_status": ((), np.int64(optimizer_status)),
            "converged": ((), bool(optimizer_status > 0)),
            "num_function_evaluations": ((), np.int64(num_function_evaluations)),
            # Placeholders: 0 is "not assessed" until the flags are defined
            "profile_quality_flag": ((), np.int64(0)),
            "h2o_vmr_quality_flag": (profile_dims, np.zeros(len(altitude), np.int64)),
        }
        attrs = {"h2o_retrieval_space": h2o_retrieval_space}
        if aerosol is not None:
            data_vars |= {
                "aerosol_extinction": (profile_dims, aerosol.extinction),
                "aerosol_extinction_prior": (profile_dims, aerosol.prior),
                "aerosol_extinction_noise_error": (profile_dims, aerosol.noise_error),
                "aerosol_extinction_solution_error": (
                    profile_dims,
                    aerosol.solution_error,
                ),
                "aerosol_extinction_averaging_kernel": (
                    kernel_dims,
                    aerosol.averaging_kernel,
                ),
            }
            attrs["aerosol_retrieval_space"] = aerosol.retrieval_space
            attrs["aerosol_extinction_wavelength"] = float(aerosol.wavelength_nm)

        self._ds = xr.Dataset(
            data_vars,
            coords={
                "time": np.datetime64(time, "ns"),
                "altitude": altitude,
                "altitude2": altitude,
            },
            attrs=attrs,
        )

    @property
    def ds(self) -> xr.Dataset:
        return self._ds


class L2DataSet:
    """An L2 granule: the profiles retrieved from one L1B granule, along time."""

    def __init__(self, ds: xr.Dataset):
        check(ds, "l2")
        self._ds = ds

    @classmethod
    def from_profiles(cls, profiles: list[L2Profile]) -> L2DataSet:
        """Stack profiles along time. They must share one altitude grid and the
        retrieval settings recorded in the global attributes."""
        if not profiles:
            msg = "An L2 granule needs at least one profile"
            raise ValueError(msg)
        datasets = [profile.ds for profile in profiles]
        if any(ds.attrs != datasets[0].attrs for ds in datasets[1:]):
            msg = "All profiles in a granule must share their retrieval settings"
            raise ValueError(msg)
        ds = xr.concat(
            datasets,
            dim="time",
            data_vars="all",
            coords="minimal",
            join="exact",
            combine_attrs="override",
        )
        return cls(apply_metadata(ds, "l2"))

    @classmethod
    def from_file(cls, file_path: Path) -> L2DataSet:
        return cls(xr.load_dataset(file_path))

    @property
    def ds(self) -> xr.Dataset:
        return self._ds

    def save(self, out_file: Path) -> None:
        self._ds.to_netcdf(out_file)
