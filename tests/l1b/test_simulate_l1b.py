from __future__ import annotations

import numpy as np

from showlib.cal_db.ideal import ideal_calibration_database
from showlib.l1b.simulate import simulate_l1b
from showlib.schemas import validate

NUM_ROWS = 3


def _geometry():
    return {
        "time": np.datetime64("2026-11-10T12:00:00"),
        "tangent_altitude": np.linspace(10e3, 20e3, NUM_ROWS),
        "tangent_latitude": np.full(NUM_ROWS, 52.0),
        "tangent_longitude": np.full(NUM_ROWS, -106.0),
        "spacecraft_latitude": 50.0,
        "spacecraft_longitude": -106.0,
        "spacecraft_altitude": 500e3,
        "solar_zenith_angle": np.full(NUM_ROWS, 60.0),
        "relative_solar_azimuth_angle": np.full(NUM_ROWS, 30.0),
        "los_azimuth_angle": np.full(NUM_ROWS, 90.0),
    }


def test_noise_free_l1b_of_a_flat_spectrum_is_that_spectrum():
    cal_db = ideal_calibration_database(NUM_ROWS, num_columns=128)
    grid = cal_db.model_wavenumber()
    radiance = np.full((1, NUM_ROWS, len(grid)), 2e-3)

    l1b = simulate_l1b(cal_db, grid, radiance, [_geometry()], noise=False)

    assert validate(l1b.ds, "l1b") == []
    np.testing.assert_allclose(l1b.ds["radiance"], 2e-3, rtol=1e-3)
    assert l1b.ds.attrs["apodization_window"] == "kaiser"


def test_noisy_l1b_reports_detector_noise():
    cal_db = ideal_calibration_database(NUM_ROWS, num_columns=128)
    grid = cal_db.model_wavenumber()
    radiance = np.full((1, NUM_ROWS, len(grid)), 2e-3)

    l1b = simulate_l1b(cal_db, grid, radiance, [_geometry()], seed=2, apodization=None)

    assert l1b.ds.attrs["radiance_noise_model"] == "detector"
    assert l1b.ds.attrs["apodization_window"] == "rectangular"
    assert np.all(l1b.ds["radiance_noise"] > 0)
    assert not np.allclose(l1b.ds["radiance"], 2e-3, rtol=1e-3)
