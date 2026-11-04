from __future__ import annotations

import numpy as np
import pytest

from showlib.cal_db import CalibrationDatabase
from showlib.l1a.data import L1ADataSet, L1AImage
from showlib.l2.lineshape import (
    SHOWLineShapes,
    apodization_from_attrs,
    instrument_line_shape,
)
from showlib.processing.l1a_to_l1b import process_l1a_to_l1b


def _non_ideal_cal_db(make_cal_db) -> CalibrationDatabase:
    num_columns = 16
    phase = np.linspace(-0.2, 0.2, 3 * (num_columns + 1)).reshape(3, num_columns + 1)
    return CalibrationDatabase(
        make_cal_db(
            num_columns=num_columns,
            littrow_edges=np.array([7330.0, 7330.1, 7330.3]),
            fringe_phase=phase,
        )
    )


def _l1a_from_spectrum(cal_db, wavenumber, spectrum) -> L1ADataSet:
    num_rows = cal_db.num_rows
    image = np.stack(
        [
            cal_db.forward_operator(row, wavenumber, np.gradient(wavenumber), 0.5)
            @ spectrum
            for row in range(num_rows)
        ]
    )
    return L1ADataSet.from_images(
        [
            L1AImage(
                image=image,
                time=np.datetime64("2026-11-04T12:00:00"),
                tangent_altitude=np.linspace(10e3, 20e3, num_rows),
                tangent_latitude=np.full(num_rows, 52.0),
                tangent_longitude=np.full(num_rows, -106.0),
                spacecraft_latitude=50.0,
                spacecraft_longitude=-106.0,
                spacecraft_altitude=500e3,
                solar_zenith_angle=np.full(num_rows, 60.0),
                relative_solar_azimuth_angle=np.full(num_rows, 30.0),
                los_azimuth_angle=np.full(num_rows, 90.0),
                exposure_time=0.5,
            )
        ]
    )


def test_line_shape_reproduces_l1a_to_l1b(make_cal_db):
    cal_db = _non_ideal_cal_db(make_cal_db)
    line_shapes = SHOWLineShapes(cal_db, apodization=10.0, model_res_cminv=0.01)
    grid = line_shapes.model_wavenumber
    spectrum = 1e-3 * (1 + 0.3 * np.sin(grid / 0.7) + 0.1 * np.cos(grid / 0.13))

    l1b = process_l1a_to_l1b(_l1a_from_spectrum(cal_db, grid, spectrum), cal_db)

    for row in range(cal_db.num_rows):
        np.testing.assert_allclose(
            l1b.ds["radiance"].isel(time=0, los=row),
            line_shapes.matrix(row) @ spectrum,
            rtol=1e-10,
        )


def test_line_shape_on_the_sample_grid_is_the_identity(make_cal_db):
    cal_db = CalibrationDatabase(make_cal_db())
    samples = cal_db.sample_wavenumber(0)
    np.testing.assert_allclose(
        instrument_line_shape(cal_db, 0, samples, apodization=None),
        np.eye(len(samples)),
        atol=1e-12,
    )


def test_line_shapes_peak_at_their_samples(make_cal_db):
    cal_db = _non_ideal_cal_db(make_cal_db)
    line_shapes = SHOWLineShapes(cal_db, apodization=10.0, model_res_cminv=0.01)

    peaks = line_shapes.model_wavenumber[np.argmax(line_shapes.matrix(1), axis=1)]

    np.testing.assert_allclose(peaks, cal_db.sample_wavenumber(1), atol=0.02)


def test_model_grid_spans_the_transmission(make_cal_db):
    cal_db = CalibrationDatabase(make_cal_db())
    line_shapes = SHOWLineShapes(cal_db, apodization=None, model_res_cminv=0.05)

    grid = line_shapes.model_wavenumber
    table = cal_db.ds["transmission_wavenumber"].to_numpy()
    np.testing.assert_allclose(np.diff(grid), 0.05)
    assert grid[0] == pytest.approx(table.min())
    assert grid[-1] == pytest.approx(table.max(), abs=0.05)


def test_matrices_are_cached_per_row(make_cal_db):
    cal_db = CalibrationDatabase(make_cal_db())
    line_shapes = SHOWLineShapes(cal_db, apodization=None, model_res_cminv=0.05)

    assert line_shapes.matrix(1) is line_shapes.matrix(1)
    np.testing.assert_allclose(
        line_shapes.matrix(1),
        instrument_line_shape(cal_db, 1, line_shapes.model_wavenumber, None),
    )


@pytest.mark.parametrize(
    ("attrs", "expected"),
    [
        ({"apodization_window": "kaiser", "apodization_beta": 7.5}, 7.5),
        ({"apodization_window": "rectangular"}, None),
    ],
)
def test_apodization_from_attrs(attrs, expected):
    assert apodization_from_attrs(attrs) == expected


def test_unknown_apodization_window_is_rejected():
    with pytest.raises(ValueError, match="apodization_window"):
        apodization_from_attrs({"apodization_window": "hann"})


def test_from_l1b_uses_its_apodization(make_cal_db):
    cal_db = CalibrationDatabase(make_cal_db())
    grid = np.linspace(7325.0, 7333.0, 200)
    l1a = _l1a_from_spectrum(cal_db, grid, np.full(grid.shape, 1e-3))
    l1b = process_l1a_to_l1b(l1a, cal_db, apodization=4.0)

    line_shapes = SHOWLineShapes.from_l1b(cal_db, l1b, model_res_cminv=0.05)

    np.testing.assert_allclose(
        line_shapes.matrix(0),
        instrument_line_shape(cal_db, 0, line_shapes.model_wavenumber, 4.0),
    )


def test_from_l1b_rejects_a_different_calibration(make_cal_db):
    cal_db = CalibrationDatabase(make_cal_db())
    grid = np.linspace(7325.0, 7333.0, 200)
    l1b = process_l1a_to_l1b(
        _l1a_from_spectrum(cal_db, grid, np.full(grid.shape, 1e-3)), cal_db
    )
    other = CalibrationDatabase(make_cal_db(sample_indices=(-4, -3, -2)))

    with pytest.raises(ValueError, match="calibration database"):
        SHOWLineShapes.from_l1b(other, l1b)
