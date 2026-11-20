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

from showlib.cal_db import CalibrationDatabase
from showlib.cal_db.ideal import ideal_calibration_database
from showlib.l1a.data import L1ADataSet
from showlib.l1a.simulate import simulate_l1a
from showlib.l1b.data import L1bDataSet
from showlib.l2.config import load_retrieval_config
from showlib.l2.data import L2DataSet
from showlib.l2.lineshape import SHOWLineShapes
from showlib.processing.l1a_to_l1b import process_l1a_to_l1b
from showlib.processing.l1b_to_l2 import process_l1b_to_l2_file
from showlib.schemas import validate

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


def _granule(tmp_path, noise_fraction=0.0, row_averaging=1):
    """An L1B granule and program of record under tmp_path; returns the L1B
    path, the calibration database's path and the config."""
    cal_db = ideal_calibration_database(len(TANGENT_ALTITUDES_M), num_columns=256)
    cal_db_path = tmp_path / "cal_db.nc"
    cal_db.ds.to_netcdf(cal_db_path)
    for folder in ["l1b", "por", "l2"]:
        (tmp_path / folder).mkdir()

    l1b_path = tmp_path / "l1b" / f"HAWC_H2OL_Radiances_L1B_{NAME}.nc"
    process_l1a_to_l1b(_l1a(cal_db), cal_db, noise_fraction=noise_fraction).save(
        l1b_path
    )
    config = load_retrieval_config(
        {
            "model_res_cminv": MODEL_RES_CMINV,
            "observation": {"upper_bound_m": 25e3, "row_averaging": row_averaging},
            "minimizer": {"max_nfev": 2},
            "preprocessor": {"wavenumber_cminv": "auto"},
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
    return l1b_path, cal_db_path, config


@pytest.mark.slow
@pytest.mark.parametrize(
    ("noise_fraction", "correlated", "row_averaging"),
    [(0.0, True, 1), (0.01, False, 1), (0.0, True, 2)],
)
def test_retrieval_runs_end_to_end(
    tmp_path, caplog, noise_fraction, correlated, row_averaging
):
    l1b_path, cal_db_path, config = _granule(tmp_path, noise_fraction, row_averaging)

    with caplog.at_level(logging.WARNING):
        process_l1b_to_l2_file(l1b_path, tmp_path / "l2", cal_db_path, config)

    l2 = L2DataSet.from_file(tmp_path / "l2" / f"HAWC_H2OL_Wvapor_L2_{NAME}.nc").ds
    assert validate(l2, "l2") == []
    assert l2.sizes["time"] == 1
    assert np.all(l2["aerosol_extinction"] > 0)
    assert int(l2["num_function_evaluations"].isel(time=0)) <= 2
    assert int(l2["num_los"].isel(time=0)) <= np.ceil(
        len(TANGENT_ALTITUDES_M) / row_averaging
    )
    assert l2.attrs["h2o_retrieval_space"] == "linear"
    assert l2.attrs["aerosol_retrieval_space"] == "log"
    assert ("Treating L1B samples as independent" in caplog.text) is not correlated


@pytest.mark.slow
def test_saved_line_shapes_give_the_same_retrieval(tmp_path):
    l1b_path, cal_db_path, config = _granule(tmp_path, row_averaging=2)
    l2_path = tmp_path / "l2" / f"HAWC_H2OL_Wvapor_L2_{NAME}.nc"
    process_l1b_to_l2_file(l1b_path, tmp_path / "l2", cal_db_path, config)
    built = L2DataSet.from_file(l2_path).ds.load()
    l2_path.unlink()

    cal_db = CalibrationDatabase.from_file(cal_db_path)
    SHOWLineShapes.from_l1b(
        cal_db, L1bDataSet.from_file(l1b_path), MODEL_RES_CMINV
    ).save(tmp_path / "line_shapes")
    line_shapes = SHOWLineShapes.load(tmp_path / "line_shapes", cal_db)
    process_l1b_to_l2_file(
        l1b_path, tmp_path / "l2", cal_db_path, config, line_shapes=line_shapes
    )

    xr.testing.assert_identical(L2DataSet.from_file(l2_path).ds, built)
