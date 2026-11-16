"""A grating tilt in the ideal calibration database: OPD and fringe phase at the
pixel corners, as described in the `model` section of cal_db.yaml."""

from __future__ import annotations

import numpy as np
import pytest

from showlib.cal_db import bin_width
from showlib.cal_db.ideal import ideal_calibration_database
from showlib.l1a.simulate import simulate_l1a
from showlib.l2.lineshape import instrument_line_shape
from showlib.processing.l1a_to_l1b import inverse_operator, process_l1a_to_l1b

NUM_ROWS, NUM_COLUMNS = 3, 64
OPD_SPACING_CM = 0.0075
ROW_PITCH_CM = 0.1
MAGNIFICATION = 2.0
TILT_RAD = 5e-4
LITTROW = 1e7 / 1362.0


def _tilted(tilt_rad=TILT_RAD, **kwargs):
    return ideal_calibration_database(
        NUM_ROWS,
        num_columns=NUM_COLUMNS,
        grating_tilt_rad=tilt_rad,
        row_pitch_cm=ROW_PITCH_CM,
        magnification=MAGNIFICATION,
        **kwargs,
    )


def _grating_heights():
    """Height of each row edge on the grating [cm], centred on zero."""
    return (np.arange(NUM_ROWS + 1) - NUM_ROWS / 2) * ROW_PITCH_CM / MAGNIFICATION


def _geometry(index=0):
    return {
        "time": np.datetime64("2026-11-10T12:00:00") + np.timedelta64(index, "s"),
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


# ------------------------------------------------------------------ builder
@pytest.mark.parametrize(("grating", "sign"), [(1, 1.0), (2, -1.0)])
def test_tilt_sets_opd_and_phase_at_the_corners(grating, sign):
    untilted = ideal_calibration_database(NUM_ROWS, num_columns=NUM_COLUMNS)
    cal_db = _tilted(tilted_grating=grating)

    offset = sign * 2 * TILT_RAD * _grating_heights()[:, np.newaxis]
    np.testing.assert_allclose(
        cal_db.ds["opd"], untilted.ds["opd"] + offset, rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(
        cal_db.ds["fringe_phase"],
        np.broadcast_to(2 * np.pi * LITTROW * offset, (NUM_ROWS + 1, NUM_COLUMNS + 1)),
    )
    assert "fringe_phase" not in untilted.ds


def test_row_edge_heights_override_the_pitch():
    heights = np.array([-0.2, -0.05, 0.0, 0.3])
    cal_db = ideal_calibration_database(
        NUM_ROWS,
        num_columns=NUM_COLUMNS,
        grating_tilt_rad=TILT_RAD,
        row_pitch_cm=99.0,
        row_edge_height_cm=heights,
    )
    untilted = ideal_calibration_database(NUM_ROWS, num_columns=NUM_COLUMNS)
    np.testing.assert_allclose(
        cal_db.ds["opd"] - untilted.ds["opd"],
        np.broadcast_to(
            2 * TILT_RAD * heights[:, np.newaxis], (NUM_ROWS + 1, NUM_COLUMNS + 1)
        ),
        atol=1e-15,
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "row_pitch_cm"),
        ({"row_pitch_cm": 0.1, "tilted_grating": 3}, "tilted_grating"),
        ({"row_pitch_cm": 0.1, "magnification": 0.0}, "magnification"),
        ({"row_edge_height_cm": [0.0, 1.0]}, "row_edge_height_cm"),
    ],
)
def test_invalid_tilt_geometry_is_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ideal_calibration_database(
            NUM_ROWS, num_columns=NUM_COLUMNS, grating_tilt_rad=TILT_RAD, **kwargs
        )


def test_scalar_fringe_visibility():
    cal_db = ideal_calibration_database(
        NUM_ROWS, num_columns=NUM_COLUMNS, fringe_visibility=0.9
    )
    np.testing.assert_allclose(cal_db.ds["fringe_visibility"], 0.9)
    assert "fringe_visibility" not in ideal_calibration_database(1, 64).ds


# ---------------------------------------------------------- fringe contrast
@pytest.mark.parametrize("tilt_rad", [TILT_RAD, 5e-3])
def test_tilt_reduces_the_contrast_by_the_sinc_factor(tilt_rad):
    """The tilt phase 2π·2εy·nu is linear along the row, so averaging over a row of
    grating height Δy multiplies the fringe by sinc(2εΔy·nu), centred on the row's
    mean height. The larger tilt changes the phase by about 17 rad along a row."""
    cal_db = _tilted(tilt_rad)
    wavenumber = np.linspace(7320.0, 7350.0, 61)
    relative = wavenumber - LITTROW
    untilted_centre = (np.arange(NUM_COLUMNS) + 0.5 - NUM_COLUMNS / 2) * OPD_SPACING_CM
    heights = _grating_heights()

    for row in range(NUM_ROWS):
        row_height = heights[row + 1] - heights[row]
        row_centre = 0.5 * (heights[row] + heights[row + 1])
        expected = (
            np.sinc(relative * OPD_SPACING_CM)
            * np.sinc(2 * tilt_rad * row_height * wavenumber)
            * np.cos(
                2 * np.pi * relative * untilted_centre[:, np.newaxis]
                + 2 * np.pi * wavenumber * 2 * tilt_rad * row_centre
            )
        )
        np.testing.assert_allclose(cal_db.fringe(row, wavenumber), expected, atol=1e-9)


# ------------------------------------------------------------ line shapes
def test_line_shape_is_diagonal_on_grid_without_apodization():
    cal_db = _tilted()
    for row in range(NUM_ROWS):
        samples = cal_db.sample_wavenumber(row)
        forward = cal_db.forward_operator(row, samples, bin_width(samples), 1.0)
        line_shape = inverse_operator(cal_db, row, apodization=None) @ forward
        normalized = line_shape / np.diag(line_shape)[:, np.newaxis]
        np.testing.assert_allclose(normalized, np.eye(len(samples)), atol=1e-9)


@pytest.mark.parametrize("apodization", [None, 10.0])
def test_line_shapes_have_unit_area(apodization):
    cal_db = _tilted()
    grid = cal_db.model_wavenumber()
    for row in range(NUM_ROWS):
        line_shape = instrument_line_shape(cal_db, row, grid, apodization)
        np.testing.assert_allclose(line_shape.sum(axis=1), 1.0, rtol=1e-10)


# -------------------------------------------------- simulate -> L1A -> L1B
def test_flat_continuum_is_recovered_through_l1a():
    cal_db = _tilted()
    grid = cal_db.model_wavenumber()
    radiance = np.full((1, NUM_ROWS, len(grid)), 2e-3)

    l1a = simulate_l1a(cal_db, grid, radiance, [_geometry()], noise=False)
    l1b = process_l1a_to_l1b(l1a, cal_db)

    np.testing.assert_allclose(l1b.ds["radiance"], 2e-3, rtol=1e-3)


def test_an_on_grid_line_appears_only_at_its_sample():
    cal_db = _tilted()
    samples = cal_db.sample_wavenumber(0)
    line = len(samples) // 2
    grid = np.union1d(cal_db.model_wavenumber(), samples)
    radiance = np.zeros((1, NUM_ROWS, len(grid)))
    radiance[0, :, np.searchsorted(grid, samples[line])] = 1.0

    l1a = simulate_l1a(cal_db, grid, radiance, [_geometry()], noise=False)
    l1b = process_l1a_to_l1b(l1a, cal_db, apodization=None)

    spectrum = l1b.ds["radiance"].isel(time=0).to_numpy()
    others = np.delete(spectrum, line, axis=1)
    assert np.all(spectrum[:, line] > 0)
    np.testing.assert_allclose(others / spectrum[:, [line]], 0.0, atol=1e-9)
