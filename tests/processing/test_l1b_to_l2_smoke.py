"""End-to-end smoke test of the L1B -> L2 retrieval wiring (run with --run-slow).

It checks that configuration, per-row line shapes, the whitened measurement and
the log-space aerosol state fit together in a real retrieval, not the quality of
the result: the spectrum is flat and the fit stops after two evaluations.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import sasktran2 as sk
import xarray as xr

from showlib.cal_db.ideal import ideal_calibration_database
from showlib.l1a.data import L1ADataSet
from showlib.l1a.simulate import simulate_l1a
from showlib.l2.config import load_retrieval_config
from showlib.l2.lineshape import SHOWLineShapes
from showlib.processing.l1a_to_l1b import process_l1a_to_l1b
from showlib.processing.l1b_to_l2 import process_l1b_to_l2_file

NAME = "20261110T1200.v0_0_1.TEST"
TANGENT_ALTITUDES_M = np.arange(10e3, 21e3, 2e3)
MODEL_RES_CMINV = 0.05


def _l1a(cal_db) -> L1ADataSet:
    grid = SHOWLineShapes(cal_db, None, MODEL_RES_CMINV).model_wavenumber
    num_rows = cal_db.num_rows
    radiance = 3e-3 * (1 - np.arange(num_rows)[:, np.newaxis] / 10) * np.ones(len(grid))
    geometry = {
        "time": np.datetime64("2026-11-10T12:00:00"),
        "tangent_altitude": TANGENT_ALTITUDES_M,
        "tangent_latitude": np.full(num_rows, 52.0),
        "tangent_longitude": np.full(num_rows, -106.0),
        "spacecraft_latitude": 50.0,
        "spacecraft_longitude": -106.0,
        "spacecraft_altitude": 500e3,
        "solar_zenith_angle": np.full(num_rows, 60.0),
        "relative_solar_azimuth_angle": np.full(num_rows, 30.0),
        "los_azimuth_angle": np.full(num_rows, 90.0),
    }
    return simulate_l1a(cal_db, grid, radiance[np.newaxis], [geometry], seed=1)


def _program_of_record(altitude_m: np.ndarray) -> xr.Dataset:
    geometry = sk.Geometry1D(
        0.5,
        0.0,
        6372e3,
        altitude_m,
        sk.InterpolationMethod.LinearInterpolation,
        sk.GeometryType.Spherical,
    )
    atmo = sk.Atmosphere(geometry, sk.Config(), wavelengths_nm=np.array([1364.0]))
    sk.climatology.us76.add_us76_standard_atmosphere(atmo)
    return xr.Dataset(
        {
            "pressure": (["time", "altitude"], atmo.pressure_pa[np.newaxis]),
            "temperature": (["time", "altitude"], atmo.temperature_k[np.newaxis]),
            "tropopause_altitude": (["time"], [11e3]),
        },
        coords={
            "altitude": altitude_m,
            "time": [np.datetime64("2026-11-10T12:00:00", "ns")],
        },
    )


@pytest.mark.slow
@pytest.mark.parametrize(("noise_fraction", "correlated"), [(0.0, True), (0.01, False)])
def test_retrieval_runs_end_to_end(tmp_path, caplog, noise_fraction, correlated):
    cal_db = ideal_calibration_database(len(TANGENT_ALTITUDES_M), num_columns=256)
    cal_db_path = tmp_path / "cal_db.nc"
    cal_db.ds.to_netcdf(cal_db_path)
    for folder in ["l1b", "por", "l2"]:
        (tmp_path / folder).mkdir()

    l1b_path = tmp_path / "l1b" / f"HAWC_H2OL_Radiances_L1B_{NAME}.nc"
    process_l1a_to_l1b(_l1a(cal_db), cal_db, noise_fraction=noise_fraction).save(
        l1b_path
    )
    samples = cal_db.sample_wavenumber(0)
    config = load_retrieval_config(
        {
            "model_res_cminv": MODEL_RES_CMINV,
            "observation": {"upper_bound_m": 25e3},
            "minimizer": {"max_nfev": 2},
            "preprocessor": {"wavenumber_cminv": float(samples[len(samples) // 2])},
        }
    )
    _program_of_record(
        np.unique(
            np.concatenate(
                [
                    np.arange(s["start"], s["stop"], s["step"])
                    for s in config["altitude_grid_m"]
                ]
            )
        ).astype(float)
    ).to_netcdf(tmp_path / "por" / f"HAWC_H2OL_Wvapor_L2_POR_{NAME}.nc")

    with caplog.at_level(logging.WARNING):
        process_l1b_to_l2_file(l1b_path, tmp_path / "l2", cal_db_path, config)

    with xr.open_dataset(tmp_path / "l2" / f"HAWC_H2OL_Wvapor_L2_{NAME}.nc") as l2:
        assert np.all(np.isfinite(l2["h2o_vmr"]))
        assert np.all(np.isfinite(l2["h2o_vmr_1sigma"]))
    assert ("Treating L1B samples as independent" in caplog.text) is not correlated
