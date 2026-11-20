from __future__ import annotations

import numpy as np
import pytest

from showlib.cal_db import CalibrationDatabase, bin_width
from showlib.cal_db.ideal import ideal_calibration_database
from showlib.l1a.data import L1ADataSet, L1AImage
from showlib.l1b.data import L1bDataSet
from showlib.l2 import lineshape
from showlib.l2.lineshape import (
    SHOWLineShapes,
    _window,
    apodization_from_attrs,
    instrument_line_shape,
)
from showlib.processing.l1a_to_l1b import inverse_operator, process_l1a_to_l1b


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
            cal_db.forward_operator(row, wavenumber, bin_width(wavenumber), 0.5)
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


def test_line_shapes_peak_at_their_samples(make_cal_db):
    cal_db = _non_ideal_cal_db(make_cal_db)
    line_shapes = SHOWLineShapes(cal_db, apodization=10.0, model_res_cminv=0.01)

    peaks = line_shapes.model_wavenumber[np.argmax(line_shapes.matrix(1), axis=1)]

    np.testing.assert_allclose(peaks, cal_db.sample_wavenumber(1), atol=0.02)


def test_model_grid_spans_the_transmission(make_cal_db):
    cal_db = CalibrationDatabase(make_cal_db())
    line_shapes = SHOWLineShapes(cal_db, apodization=None, model_res_cminv=0.05)

    np.testing.assert_array_equal(
        line_shapes.model_wavenumber, cal_db.model_wavenumber(resolution_cminv=0.05)
    )


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


def test_instrument_model_selects_requested_samples_per_row(make_cal_db):
    cal_db = _non_ideal_cal_db(make_cal_db)
    line_shapes = SHOWLineShapes(cal_db, apodization=None, model_res_cminv=0.05)
    samples = cal_db.sample_wavenumber(0)

    model = line_shapes.instrument_model(rows=[1, 0], sample_wavenumber=samples[[2, 1]])

    np.testing.assert_allclose(model.sample_wavenumber, samples[[1, 2]])
    np.testing.assert_allclose(model.matrices[0], line_shapes.matrix(1)[[1, 2]])
    np.testing.assert_allclose(model.matrices[1], line_shapes.matrix(0)[[1, 2]])
    np.testing.assert_allclose(model.model_wavenumber, line_shapes.model_wavenumber)


def test_instrument_model_rejects_unknown_samples(make_cal_db):
    line_shapes = SHOWLineShapes(
        CalibrationDatabase(make_cal_db()), apodization=None, model_res_cminv=0.05
    )
    with pytest.raises(ValueError, match="sample"):
        line_shapes.instrument_model(rows=[0], sample_wavenumber=[7000.0])


def test_instrument_model_matches_samples_on_a_fine_grid(make_cal_db):
    """Samples 0.05 cm-1 apart must not be confused with their neighbours."""
    cal_db = CalibrationDatabase(
        make_cal_db(num_columns=200, sample_indices=(-4, -3, -2, -1))
    )
    line_shapes = SHOWLineShapes(cal_db, apodization=None, model_res_cminv=0.05)
    samples = cal_db.sample_wavenumber(0)

    model = line_shapes.instrument_model(rows=[0], sample_wavenumber=samples[[2]])

    np.testing.assert_allclose(model.matrices[0], line_shapes.matrix(0)[[2]])


def _three_row_l1b(make_cal_db):
    cal_db = CalibrationDatabase(make_cal_db(num_rows=3))
    grid = np.linspace(7325.0, 7333.0, 200)
    l1b = process_l1a_to_l1b(
        _l1a_from_spectrum(cal_db, grid, np.full(grid.shape, 1e-3)), cal_db
    )
    return cal_db, l1b


def test_from_l1b_accepts_a_subset_of_detector_rows(make_cal_db):
    cal_db, l1b = _three_row_l1b(make_cal_db)
    subset = L1bDataSet(l1b.ds.isel(los=[0, 2]))
    SHOWLineShapes.from_l1b(cal_db, subset, model_res_cminv=0.05)


def test_from_l1b_rejects_rows_the_calibration_does_not_have(make_cal_db):
    cal_db, l1b = _three_row_l1b(make_cal_db)
    shifted = L1bDataSet(l1b.ds.assign_coords(los=l1b.ds["los"] + 2))
    with pytest.raises(ValueError, match="detector rows"):
        SHOWLineShapes.from_l1b(cal_db, shifted)


@pytest.mark.parametrize("apodization", [None, 10.0])
def test_line_shapes_have_unit_area(make_cal_db, apodization):
    cal_db = _non_ideal_cal_db(make_cal_db)
    grid = cal_db.model_wavenumber()
    line_shape = instrument_line_shape(cal_db, 1, grid, apodization)
    np.testing.assert_allclose(line_shape.sum(axis=1), 1.0, rtol=1e-10)


def test_from_l1b_rejects_a_slightly_shifted_grid(make_cal_db):
    cal_db, l1b = _three_row_l1b(make_cal_db)
    shifted = L1bDataSet(l1b.ds.assign_coords(wavenumber=l1b.ds["wavenumber"] + 0.01))
    with pytest.raises(ValueError, match="does not match"):
        SHOWLineShapes.from_l1b(cal_db, shifted)


# ---------------------------------------------------------------- row groups
def test_a_row_group_has_the_mean_line_shape(make_cal_db):
    cal_db = _non_ideal_cal_db(make_cal_db)
    line_shapes = SHOWLineShapes(cal_db, apodization=10.0, model_res_cminv=0.05)

    np.testing.assert_allclose(
        line_shapes.matrix((0, 1)),
        0.5 * (line_shapes.matrix(0) + line_shapes.matrix(1)),
    )
    np.testing.assert_array_equal(line_shapes.matrix((1,)), line_shapes.matrix(1))
    np.testing.assert_array_equal(
        line_shapes.matrix(np.int64(1)), line_shapes.matrix(1)
    )


def test_a_row_group_has_the_noise_correlation_of_a_mean(make_cal_db):
    cal_db = _non_ideal_cal_db(make_cal_db)
    line_shapes = SHOWLineShapes(cal_db, apodization=10.0, model_res_cminv=0.05)

    operators = [inverse_operator(cal_db, row, 10.0) for row in (0, 1)]
    covariance = sum(operator @ operator.T for operator in operators) / 4
    sigma = np.sqrt(np.diag(covariance))
    np.testing.assert_allclose(
        line_shapes.noise_correlation((0, 1)), covariance / np.outer(sigma, sigma)
    )
    samples = cal_db.sample_wavenumber(0)[[2, 0]]
    np.testing.assert_allclose(
        line_shapes.noise_correlation((0, 1), samples),
        line_shapes.noise_correlation((0, 1))[np.ix_([2, 0], [2, 0])],
    )


def test_instrument_model_mixes_rows_and_groups(make_cal_db):
    cal_db = _non_ideal_cal_db(make_cal_db)
    line_shapes = SHOWLineShapes(cal_db, apodization=None, model_res_cminv=0.05)
    samples = cal_db.sample_wavenumber(0)

    model = line_shapes.instrument_model([(0, 1), 1], samples)

    np.testing.assert_allclose(model.matrices[0], line_shapes.matrix((0, 1)))
    np.testing.assert_allclose(model.matrices[1], line_shapes.matrix(1))


def test_rows_with_different_samples_cannot_be_averaged(make_cal_db):
    ds = make_cal_db(num_rows=2)
    ds["sample_wavenumber"][1] = ds["sample_wavenumber"][1] - 0.625
    line_shapes = SHOWLineShapes(CalibrationDatabase(ds), apodization=None)
    with pytest.raises(ValueError, match="different sample wavenumbers"):
        line_shapes.matrix((0, 1))
    with pytest.raises(ValueError, match="different sample wavenumbers"):
        line_shapes.noise_correlation((0, 1))


# ---------------------------------------------------------- windowed models
def test_inverse_is_shared_by_line_shapes_and_noise(make_cal_db, monkeypatch):
    calls = []
    original = lineshape.inverse_operator

    def counting(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(lineshape, "inverse_operator", counting)
    line_shapes = SHOWLineShapes(
        _non_ideal_cal_db(make_cal_db), apodization=10.0, model_res_cminv=0.05
    )
    line_shapes.matrix(0)
    line_shapes.noise_correlation(0)
    line_shapes.matrix((0, 1))
    line_shapes.noise_correlation((0, 1))

    assert sorted(calls) == [0, 1]


@pytest.mark.slow
def test_a_windowed_model_keeps_one_samples_line_shape():
    cal_db = ideal_calibration_database(2)
    line_shapes = SHOWLineShapes(cal_db, apodization=10.0, model_res_cminv=0.01)
    sample = cal_db.sample_wavenumber(0)[100]
    grid = line_shapes.model_wavenumber
    full = line_shapes.instrument_model([0, (0, 1)], [sample])

    model = line_shapes.instrument_model([0, (0, 1)], [sample], tolerance=1e-3)

    assert len(model.model_wavenumber) < len(grid) / 10
    assert model.model_wavenumber[0] < sample < model.model_wavenumber[-1]
    window = np.isin(grid, model.model_wavenumber)
    np.testing.assert_array_equal(grid[window], model.model_wavenumber)
    # Each line shape keeps its area, and nearly all of its weight is inside
    np.testing.assert_allclose(
        model.matrices.sum(axis=-1), full.matrices.sum(axis=-1), rtol=1e-12
    )
    outside = np.abs(full.matrices[..., ~window]).sum(axis=-1)
    assert np.all(outside <= 1e-3 * np.abs(full.matrices).sum(axis=-1))
    # A smooth spectrum is modelled as on the full grid
    spectrum = 1e-3 * (1 + 0.2 * np.sin(grid / 3.0))
    np.testing.assert_allclose(
        model.matrices @ spectrum[window], full.matrices @ spectrum, rtol=1e-3
    )


def test_without_a_tolerance_the_model_grid_is_the_whole_grid(make_cal_db):
    line_shapes = SHOWLineShapes(
        _non_ideal_cal_db(make_cal_db), apodization=10.0, model_res_cminv=0.05
    )
    samples = line_shapes.cal_db.sample_wavenumber(0)
    model = line_shapes.instrument_model([0], samples[[1]])
    np.testing.assert_array_equal(model.model_wavenumber, line_shapes.model_wavenumber)


def test_window_is_the_union_over_line_shapes():
    line_shapes = np.zeros((2, 1, 10))
    line_shapes[0, 0, 2:4] = 1.0
    line_shapes[1, 0, 5:8] = [1.0, -1.0, 1.0]
    assert _window(line_shapes, 1e-3) == slice(2, 8)
    # Within the tolerance, a small tail is dropped
    line_shapes[1, 0, 9] = 1e-4
    assert _window(line_shapes, 1e-3) == slice(2, 8)
    assert _window(line_shapes, 1e-5) == slice(2, 10)


# ------------------------------------------------------------- saved files
def test_saved_line_shapes_load_unchanged(make_cal_db, tmp_path):
    cal_db = _non_ideal_cal_db(make_cal_db)
    built = SHOWLineShapes(cal_db, apodization=10.0, model_res_cminv=0.05)
    built.save(tmp_path / "line_shapes")

    loaded = SHOWLineShapes.load(tmp_path / "line_shapes", cal_db)

    assert loaded.apodization == 10.0
    assert loaded.model_res_cminv == 0.05
    np.testing.assert_array_equal(loaded.model_wavenumber, built.model_wavenumber)
    for row in [0, 1, (0, 1)]:
        np.testing.assert_array_equal(loaded.matrix(row), built.matrix(row))
        np.testing.assert_array_equal(
            loaded.noise_correlation(row), built.noise_correlation(row)
        )
    samples = cal_db.sample_wavenumber(0)[[1, 2]]
    np.testing.assert_array_equal(
        loaded.instrument_model([1, 0], samples, tolerance=1e-3).matrices,
        built.instrument_model([1, 0], samples, tolerance=1e-3).matrices,
    )


def test_loaded_line_shapes_are_memory_mapped_not_rebuilt(
    make_cal_db, tmp_path, monkeypatch
):
    cal_db = _non_ideal_cal_db(make_cal_db)
    SHOWLineShapes(cal_db, apodization=None, model_res_cminv=0.05).save(
        tmp_path / "line_shapes"
    )

    def fail(*args, **kwargs):
        pytest.fail("rebuilt B")

    monkeypatch.setattr(lineshape, "inverse_operator", fail)
    loaded = SHOWLineShapes.load(tmp_path / "line_shapes", cal_db)

    assert isinstance(loaded.matrix(0).base, np.memmap)
    assert not loaded.matrix(0).flags.writeable
    loaded.noise_correlation((0, 1))


def test_saving_does_not_overwrite(make_cal_db, tmp_path):
    line_shapes = SHOWLineShapes(
        CalibrationDatabase(make_cal_db()), apodization=None, model_res_cminv=0.05
    )
    line_shapes.save(tmp_path / "line_shapes")

    with pytest.raises(FileExistsError):
        line_shapes.save(tmp_path / "line_shapes")


def test_loading_rejects_a_different_calibration(make_cal_db, tmp_path):
    ds = make_cal_db()
    SHOWLineShapes(
        CalibrationDatabase(ds), apodization=None, model_res_cminv=0.05
    ).save(tmp_path / "line_shapes")
    changed = ds.copy(deep=True)
    changed["pixel_etendue"] = changed["pixel_etendue"] * 1.001

    with pytest.raises(ValueError, match="calibration database"):
        SHOWLineShapes.load(tmp_path / "line_shapes", CalibrationDatabase(changed))


def test_fingerprint_follows_the_calibration_contents(make_cal_db):
    ds = make_cal_db()
    same = ds.copy(deep=True)
    same.attrs["history"] = "re-saved"
    changed = ds.copy(deep=True)
    changed["pixel_etendue"] = changed["pixel_etendue"] * 1.001

    fingerprint = CalibrationDatabase(ds).fingerprint()
    assert CalibrationDatabase(same).fingerprint() == fingerprint
    assert CalibrationDatabase(changed).fingerprint() != fingerprint


def test_check_l1b_rejects_a_different_apodization_or_grid(make_cal_db):
    cal_db = CalibrationDatabase(make_cal_db())
    grid = np.linspace(7325.0, 7333.0, 200)
    l1b = process_l1a_to_l1b(
        _l1a_from_spectrum(cal_db, grid, np.full(grid.shape, 1e-3)),
        cal_db,
        apodization=4.0,
    )
    line_shapes = SHOWLineShapes(cal_db, apodization=4.0, model_res_cminv=0.05)
    line_shapes.check_l1b(cal_db, l1b, model_res_cminv=0.05)

    with pytest.raises(ValueError, match="apodization"):
        SHOWLineShapes(cal_db, apodization=10.0, model_res_cminv=0.05).check_l1b(
            cal_db, l1b, model_res_cminv=0.05
        )
    with pytest.raises(ValueError, match="model_res_cminv"):
        line_shapes.check_l1b(cal_db, l1b, model_res_cminv=0.01)
    other = CalibrationDatabase(make_cal_db(dark_offset=1.0))
    with pytest.raises(ValueError, match="calibration database"):
        line_shapes.check_l1b(other, l1b, model_res_cminv=0.05)
