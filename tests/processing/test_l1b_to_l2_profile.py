"""Mapping SHOWRetrieval results onto an L2 profile, without running a retrieval."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from scipy.optimize import OptimizeResult

from showlib.l2.data import L2DataSet
from showlib.processing.l1b_to_l2 import _profile
from showlib.schemas import validate

ALTITUDE = np.array([5e3, 10e3, 15e3])
AEROSOL = "stratospheric_aerosol_extinction_per_m"


def _state(error_scale: float) -> xr.Dataset:
    """A described state, with every 1-sigma error scaled by error_scale."""
    profile, kernel = ["altitude"], ["altitude", "altitude_2"]
    return xr.Dataset(
        {
            "h2o_vmr": (profile, [6e-6, 5e-6, 4e-6]),
            "h2o_vmr_prior": (profile, [5e-6, 5e-6, 5e-6]),
            "h2o_vmr_1sigma_error": (profile, error_scale * np.full(3, 1e-7)),
            "h2o_vmr_averaging_kernel": (kernel, 0.9 * np.eye(3)),
            "lambertian_albedo": ("wavelengths_nm", [0.28, 0.32]),
            "lambertian_albedo_1sigma_error": (
                "wavelengths_nm",
                error_scale * np.array([0.01, 0.03]),
            ),
            AEROSOL: (profile, [1e-6, 2e-6, 3e-6]),
            f"{AEROSOL}_prior": (profile, [1e-6, 1e-6, 1e-6]),
            f"{AEROSOL}_1sigma_error": (profile, error_scale * np.full(3, 1e-8)),
            f"{AEROSOL}_averaging_kernel": (kernel, 0.5 * np.eye(3)),
        },
        coords={
            "altitude": ALTITUDE,
            "altitude_2": ALTITUDE,
            "wavelengths_nm": [1360.0, 1368.0],
        },
    )


def _results(status: int = 2) -> dict:
    return {
        "state": _state(1.0),
        "solution_state": _state(3.0),
        "minimizer": {"minimizer": OptimizeResult(cost=1.2, status=status, nfev=9)},
    }


def _state_kwargs(aerosol_enabled: bool = True) -> dict:
    return {
        "absorbers": {"h2o": {"log_space": False}},
        "aerosols": {
            "stratospheric_aerosol": {
                "enabled": aerosol_enabled,
                "log_space": True,
                "nominal_wavelength": 1350,
            }
        },
    }


PROFILE_KWARGS = {
    "time": np.datetime64("2026-11-10T12:00:00"),
    "latitude": 52.0,
    "longitude": -106.0,
    "tropopause_altitude": 11e3,
    "lower_altitude_limit": 6e3,
    "upper_altitude_limit": 25e3,
    "num_los": 10,
}


def test_noise_and_solution_errors_come_from_their_states():
    ds = _profile(_results(), _state_kwargs(), **PROFILE_KWARGS).ds

    np.testing.assert_allclose(ds["h2o_vmr_noise_error"], 1e-7)
    np.testing.assert_allclose(ds["h2o_vmr_solution_error"], 3e-7)
    np.testing.assert_allclose(ds["aerosol_extinction_noise_error"], 1e-8)
    np.testing.assert_allclose(ds["aerosol_extinction_solution_error"], 3e-8)
    np.testing.assert_allclose(ds["h2o_vmr_averaging_kernel"], 0.9 * np.eye(3))
    np.testing.assert_allclose(
        ds["aerosol_extinction_averaging_kernel"], 0.5 * np.eye(3)
    )


def test_the_albedo_is_not_stored():
    ds = _profile(_results(), _state_kwargs(), **PROFILE_KWARGS).ds
    assert not [name for name in ds if "albedo" in name]


@pytest.mark.parametrize(("status", "converged"), [(2, True), (0, False), (-1, False)])
def test_diagnostics_and_settings_are_recorded(status, converged):
    ds = _profile(_results(status), _state_kwargs(), **PROFILE_KWARGS).ds

    assert float(ds["cost"]) == 1.2
    assert int(ds["optimizer_status"]) == status
    assert bool(ds["converged"]) is converged
    assert int(ds["num_function_evaluations"]) == 9
    assert int(ds["num_los"]) == 10
    assert ds.attrs == {
        "h2o_retrieval_space": "linear",
        "aerosol_retrieval_space": "log",
        "aerosol_extinction_wavelength": 1350.0,
    }
    l2 = L2DataSet.from_profiles(
        [_profile(_results(status), _state_kwargs(), **PROFILE_KWARGS)]
    )
    assert validate(l2.ds, "l2") == []


def test_a_disabled_aerosol_is_left_out():
    ds = _profile(_results(), _state_kwargs(aerosol_enabled=False), **PROFILE_KWARGS).ds
    assert "aerosol_extinction" not in ds
    assert "aerosol_retrieval_space" not in ds.attrs
