from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from showlib.l2.data import AerosolProfile, L2DataSet, L2Profile
from showlib.schemas import SchemaValidationError, validate

ALTITUDES = np.arange(5000.0, 30001.0, 5000.0)


def _aerosol(wavelength_nm: float = 1350.0) -> AerosolProfile:
    n = len(ALTITUDES)
    return AerosolProfile(
        extinction=np.full(n, 1e-6),
        prior=np.full(n, 2e-6),
        noise_error=np.full(n, 1e-7),
        solution_error=np.full(n, 3e-7),
        averaging_kernel=0.5 * np.eye(n),
        retrieval_space="log",
        wavelength_nm=wavelength_nm,
    )


def _profile(t: str, scale: float = 1.0, **overrides) -> L2Profile:
    n = len(ALTITUDES)
    kwargs = {
        "time": pd.Timestamp(t),
        "latitude": 40.0,
        "longitude": -100.0,
        "altitude": ALTITUDES,
        "h2o_vmr": np.full(n, 5e-6) * scale,
        "h2o_vmr_prior": np.full(n, 4e-6),
        "h2o_vmr_noise_error": np.full(n, 1e-7),
        "h2o_vmr_solution_error": np.full(n, 2e-7),
        "h2o_vmr_averaging_kernel": np.eye(n),
        "h2o_retrieval_space": "linear",
        "tropopause_altitude": 11500.0,
        "lower_altitude_limit": 8000.0,
        "upper_altitude_limit": 30000.0,
        "num_los": 12,
        "cost": 1.1,
        "optimizer_status": 2,
        "num_function_evaluations": 7,
        "aerosol": _aerosol(),
    }
    return L2Profile(**(kwargs | overrides))


def test_a_profile_is_the_granule_at_one_time():
    ds = _profile("2023-11-28T17:00:00").ds

    for var in ["h2o_vmr", "h2o_vmr_noise_error", "aerosol_extinction"]:
        assert ds[var].dims == ("altitude",)
    assert ds["h2o_vmr_averaging_kernel"].dims == ("altitude", "altitude2")
    np.testing.assert_array_equal(ds["altitude2"], ALTITUDES)
    assert bool(ds["converged"])
    assert int(ds["profile_quality_flag"]) == 0
    np.testing.assert_array_equal(ds["h2o_vmr_quality_flag"], 0)
    assert "surface_albedo" not in ds
    assert ds.attrs == {
        "h2o_retrieval_space": "linear",
        "aerosol_retrieval_space": "log",
        "aerosol_extinction_wavelength": 1350.0,
    }


def test_granule_round_trip(tmp_path):
    profiles = [
        _profile("2023-11-28T17:00:00", scale=1.0),
        _profile("2023-11-28T17:01:00", scale=2.0, optimizer_status=0),
    ]
    out_file = tmp_path / "l2.nc"
    L2DataSet.from_profiles(profiles).save(out_file)

    ds = L2DataSet.from_file(out_file).ds
    assert validate(ds, "l2") == []
    assert ds.sizes == {
        "time": 2,
        "altitude": len(ALTITUDES),
        "altitude2": len(ALTITUDES),
    }
    assert ds["h2o_vmr"].dims == ("time", "altitude")
    assert ds["h2o_vmr_averaging_kernel"].dims == ("time", "altitude", "altitude2")
    np.testing.assert_allclose(ds["h2o_vmr"].isel(time=1), 1e-5)
    np.testing.assert_array_equal(ds["converged"], [True, False])
    assert ds["h2o_vmr"].attrs["units"] == "1"
    assert ds.attrs["schema_version"]
    assert pd.Timestamp(ds.time.to_numpy()[1]) == pd.Timestamp("2023-11-28T17:01:00")


def test_aerosol_is_optional():
    ds = L2DataSet.from_profiles([_profile("2023-11-28T17:00:00", aerosol=None)]).ds
    assert "aerosol_extinction" not in ds
    assert "aerosol_retrieval_space" not in ds.attrs


def test_profiles_must_share_their_retrieval_settings():
    profiles = [
        _profile("2023-11-28T17:00:00"),
        _profile("2023-11-28T17:01:00", aerosol=_aerosol(wavelength_nm=745.0)),
    ]
    with pytest.raises(ValueError, match="retrieval settings"):
        L2DataSet.from_profiles(profiles)


def test_profiles_must_share_their_altitude_grid():
    shifted = _profile("2023-11-28T17:01:00", altitude=ALTITUDES + 1.0)
    with pytest.raises(ValueError, match="altitude"):
        L2DataSet.from_profiles([_profile("2023-11-28T17:00:00"), shifted])


def test_an_empty_granule_is_rejected():
    with pytest.raises(ValueError, match="at least one profile"):
        L2DataSet.from_profiles([])


def test_invalid_granules_are_rejected():
    ds = L2DataSet.from_profiles([_profile("2023-11-28T17:00:00")]).ds
    with pytest.raises(SchemaValidationError, match="h2o_vmr_solution_error"):
        L2DataSet(ds.drop_vars("h2o_vmr_solution_error"))
