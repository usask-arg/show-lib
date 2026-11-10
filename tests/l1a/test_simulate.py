from __future__ import annotations

import numpy as np
import pytest

from showlib.cal_db import bin_width
from showlib.cal_db.ideal import ideal_calibration_database
from showlib.l1a.simulate import simulate_l1a
from showlib.l2.lineshape import SHOWLineShapes
from showlib.schemas import validate

NUM_ROWS = 4


@pytest.fixture(scope="module")
def cal_db():
    return ideal_calibration_database(NUM_ROWS, num_columns=64)


@pytest.fixture(scope="module")
def wavenumber(cal_db):
    return SHOWLineShapes(cal_db, None, model_res_cminv=0.05).model_wavenumber


def _geometry(index=0, num_rows=NUM_ROWS):
    return {
        "time": np.datetime64("2026-11-10T12:00:00") + np.timedelta64(index, "s"),
        "tangent_altitude": np.linspace(10e3, 25e3, num_rows),
        "tangent_latitude": np.full(num_rows, 52.0),
        "tangent_longitude": np.full(num_rows, -106.0),
        "spacecraft_latitude": 50.0,
        "spacecraft_longitude": -106.0,
        "spacecraft_altitude": 500e3,
        "solar_zenith_angle": np.full(num_rows, 60.0),
        "relative_solar_azimuth_angle": np.full(num_rows, 30.0),
        "los_azimuth_angle": np.full(num_rows, 90.0),
    }


def _radiance(wavenumber, num_images=1, num_rows=NUM_ROWS):
    rows = np.arange(num_rows)[:, np.newaxis]
    spectrum = 2e-3 * (1 + 0.2 * np.sin(wavenumber / 0.3))[np.newaxis, :]
    single = spectrum * (1 - 0.1 * rows)
    return np.stack([single * (1 + 0.05 * i) for i in range(num_images)])


def _expected_counts(cal_db, wavenumber, radiance, exposure_time):
    widths = bin_width(wavenumber)
    return np.stack(
        [
            cal_db.forward_operator(row, wavenumber, widths, exposure_time)
            @ radiance[row]
            for row in range(cal_db.num_rows)
        ]
    )


def test_noise_free_image_is_the_forward_model(cal_db, wavenumber):
    radiance = _radiance(wavenumber)

    l1a = simulate_l1a(
        cal_db, wavenumber, radiance, [_geometry()], exposure_time=0.05, noise=False
    )

    expected = _expected_counts(cal_db, wavenumber, radiance[0], 0.05)
    np.testing.assert_allclose(l1a.ds["image"].isel(time=0), expected, rtol=1e-12)
    np.testing.assert_allclose(l1a.simulation["image_noiseless"].isel(time=0), expected)
    assert not bool(l1a.simulation["noise_enabled"])
    np.testing.assert_allclose(
        l1a.ds["tangent_altitude"].isel(time=0), _geometry()["tangent_altitude"]
    )
    assert validate(l1a.ds, "l1a") == []


def test_auto_exposure_puts_the_brightest_pixel_at_the_full_well_fraction(
    cal_db, wavenumber
):
    l1a = simulate_l1a(
        cal_db, wavenumber, _radiance(wavenumber), [_geometry()], noise=False
    )

    gain = float(cal_db.ds["detector_gain"])
    full_well = float(cal_db.ds["detector_full_well"])
    peak = float(l1a.ds["image"].max()) * gain / full_well
    np.testing.assert_allclose(peak, 0.8)
    np.testing.assert_allclose(l1a.simulation["expected_peak_full_well_fraction"], 0.8)
    np.testing.assert_allclose(l1a.simulation["auto_exposure_full_well_fraction"], 0.8)
    assert float(l1a.ds["exposure_time"].isel(time=0)) > 0


def test_noise_matches_the_l1a_to_l1b_noise_model(cal_db, wavenumber):
    """Shot, read and quantization noise as process_l1a_to_l1b propagates it."""
    num_exposures = 4
    radiance = np.repeat(_radiance(wavenumber), 50, axis=0)
    l1a = simulate_l1a(
        cal_db,
        wavenumber,
        radiance,
        [_geometry(i) for i in range(50)],
        num_averaged_exposures=num_exposures,
        seed=3,
    )

    gain = float(cal_db.ds["detector_gain"])
    read_noise = float(cal_db.ds["detector_read_noise"])
    expected = l1a.simulation["image_noiseless"].to_numpy()
    variance = (expected * gain + read_noise**2 + gain**2 / 12) / (
        num_exposures * gain**2
    )
    z = (l1a.ds["image"].to_numpy() - expected) / np.sqrt(variance)
    assert abs(z.mean()) < 0.02
    assert abs(z.std() - 1) < 0.02
    assert int(l1a.ds["num_averaged_exposures"].isel(time=0)) == num_exposures
    assert int(l1a.simulation["noise_seed"]) == 3


def test_saturation_is_flagged_and_clipped(cal_db, wavenumber):
    l1a = simulate_l1a(
        cal_db,
        wavenumber,
        _radiance(wavenumber),
        [_geometry()],
        exposure_time=5.0,
        seed=0,
    )

    full_well_counts = float(cal_db.ds["detector_full_well"]) / float(
        cal_db.ds["detector_gain"]
    )
    assert bool(l1a.ds["saturated"].any())
    assert float(l1a.ds["image"].max()) <= full_well_counts


def test_the_seed_makes_runs_reproducible(cal_db, wavenumber):
    def run(seed):
        return simulate_l1a(
            cal_db, wavenumber, _radiance(wavenumber), [_geometry()], seed=seed
        ).ds["image"]

    np.testing.assert_array_equal(run(5), run(5))
    assert not np.array_equal(run(5), run(6))


def test_images_are_stacked_along_time(cal_db, wavenumber):
    l1a = simulate_l1a(
        cal_db,
        wavenumber,
        _radiance(wavenumber, num_images=3),
        [_geometry(i) for i in range(3)],
        noise=False,
    )
    assert l1a.ds.sizes["time"] == 3
    assert l1a.simulation["image_noiseless"].dims == ("time", "los", "pixelcolumn")


def test_radiance_must_match_the_detector_and_grid(cal_db, wavenumber):
    with pytest.raises(ValueError, match="radiance"):
        simulate_l1a(
            cal_db,
            wavenumber,
            _radiance(wavenumber, num_rows=NUM_ROWS + 1),
            [_geometry()],
        )
    with pytest.raises(ValueError, match="geometry"):
        simulate_l1a(cal_db, wavenumber, _radiance(wavenumber), [])


def test_noise_needs_the_detector_limits(cal_db, wavenumber):
    without_full_well = type(cal_db)(cal_db.ds.drop_vars("detector_full_well"))
    with pytest.raises(ValueError, match="detector_full_well"):
        simulate_l1a(
            without_full_well, wavenumber, _radiance(wavenumber), [_geometry()]
        )


def test_noise_free_images_are_clipped_and_flagged(cal_db, wavenumber):
    l1a = simulate_l1a(
        cal_db,
        wavenumber,
        _radiance(wavenumber),
        [_geometry()],
        exposure_time=5.0,
        noise=False,
    )

    full_well_counts = float(cal_db.ds["detector_full_well"]) / float(
        cal_db.ds["detector_gain"]
    )
    assert float(l1a.ds["image"].max()) <= full_well_counts
    assert bool(l1a.ds["saturated"].any())
    assert float(l1a.simulation["image_noiseless"].max()) > full_well_counts
