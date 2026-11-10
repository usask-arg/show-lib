from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from showlib.cal_db import CalibrationDatabase, bin_width
from showlib.schemas import SchemaValidationError

WAVENUMBER = np.linspace(7327.0, 7333.0, 25)


def _brute_force_fringe(ds, row, column, wavenumber, scale=1.0, phase_offset=0.0):
    """Average the fringe over a dense midpoint grid across the pixel area."""
    num = 400
    t = (np.arange(num) + 0.5) / num
    u, v = np.meshgrid(t, t, indexing="ij")
    opd = ds["opd"].to_numpy()[row : row + 2, column : column + 2]
    phase = (
        ds["fringe_phase"].to_numpy()[row : row + 2, column : column + 2]
        if "fringe_phase" in ds
        else np.zeros((2, 2))
    )
    littrow = ds["littrow_wavenumber"].to_numpy()[row : row + 2]

    def bilinear(corners):
        return (
            (1 - v) * (1 - u) * corners[0, 0]
            + (1 - v) * u * corners[0, 1]
            + v * (1 - u) * corners[1, 0]
            + v * u * corners[1, 1]
        )

    d = bilinear(opd)
    phi = bilinear(phase)
    sigma = (1 - v) * littrow[0] + v * littrow[1]
    return np.array(
        [
            np.mean(np.cos(2 * np.pi * (nu - sigma) * scale * d + phi + phase_offset))
            for nu in wavenumber
        ]
    )


def test_invalid_dataset_is_rejected(make_cal_db):
    with pytest.raises(SchemaValidationError, match=r"variables\.opd"):
        CalibrationDatabase(make_cal_db().drop_vars("opd"))


def test_from_file(make_cal_db, tmp_path):
    path = tmp_path / "cal_db.nc"
    make_cal_db().to_netcdf(path)
    cal_db = CalibrationDatabase.from_file(path)
    assert cal_db.num_rows == 2
    assert cal_db.num_columns == 16


def test_sample_wavenumber(make_cal_db):
    cal_db = CalibrationDatabase(make_cal_db(sample_indices=(-2, -1)))
    np.testing.assert_allclose(cal_db.sample_wavenumber(1), [7328.75, 7329.375])


def test_pixel_centre_opd_is_mean_of_corners(make_cal_db):
    ds = make_cal_db()
    ds["opd"] = ds["opd"] * np.array([[1.0], [1.1], [1.2]])
    expected = ds["opd"].to_numpy()[1:3].mean(axis=0)
    expected = 0.5 * (expected[:-1] + expected[1:])
    np.testing.assert_allclose(CalibrationDatabase(ds).pixel_centre_opd(1), expected)


def test_fringe_reduces_to_sinc_cos_for_an_ideal_pixel(make_cal_db):
    ds = make_cal_db()
    cal_db = CalibrationDatabase(ds)
    centre = cal_db.pixel_centre_opd(0)
    relative = WAVENUMBER - 7330.0
    expected = np.sinc(relative * 0.1) * np.cos(
        2 * np.pi * relative * centre[:, np.newaxis]
    )
    np.testing.assert_allclose(cal_db.fringe(0, WAVENUMBER), expected, atol=1e-12)


def test_fringe_averages_tilt_phase_and_smile_over_the_pixel(make_cal_db):
    num_rows, num_columns = 2, 4
    base = (np.arange(num_columns + 1) - num_columns / 2) * 0.1
    opd = base + np.array([[0.0], [0.03], [0.07]])
    phase = np.linspace(0, 0.8, (num_rows + 1) * (num_columns + 1)).reshape(opd.shape)
    ds = make_cal_db(
        num_rows=num_rows,
        num_columns=num_columns,
        opd=opd,
        littrow_edges=np.array([7330.0, 7330.4, 7330.6]),
        fringe_phase=phase,
    )
    cal_db = CalibrationDatabase(ds)

    result = cal_db.fringe(1, WAVENUMBER, num_row_nodes=12)

    for column in range(num_columns):
        np.testing.assert_allclose(
            result[column],
            _brute_force_fringe(ds, 1, column, WAVENUMBER),
            atol=1e-5,
        )


def test_fringe_includes_dispersion(make_cal_db):
    reference = 7330.0
    grid = np.linspace(7320.0, 7340.0, 41)
    offset = grid - reference
    ds = make_cal_db().assign(
        fringe_frequency_scale=("dispersion_wavenumber", 1 + 1e-3 * offset**2),
        dispersion_phase=("dispersion_wavenumber", 0.02 * offset**2),
        dispersion_reference_wavenumber=((), reference),
    )
    ds = ds.assign_coords(dispersion_wavenumber=grid)
    for name, units in [
        ("fringe_frequency_scale", "1"),
        ("dispersion_phase", "rad"),
        ("dispersion_reference_wavenumber", "cm-1"),
        ("dispersion_wavenumber", "cm-1"),
    ]:
        ds[name].attrs["units"] = units
    cal_db = CalibrationDatabase(ds)

    nu = np.array([7327.5, 7331.0])
    scale = np.interp(nu, grid, 1 + 1e-3 * offset**2)
    phase = np.interp(nu, grid, 0.02 * offset**2)
    result = cal_db.fringe(0, nu)

    for column in range(cal_db.num_columns):
        for j in range(len(nu)):
            expected = _brute_force_fringe(
                ds, 0, column, nu[j : j + 1], scale=scale[j], phase_offset=phase[j]
            )
            np.testing.assert_allclose(result[column, j], expected, atol=1e-5)


def test_fringe_visibility_scales_the_fringe(make_cal_db):
    visibility = np.full((2, 16), 0.5)
    ideal = CalibrationDatabase(make_cal_db())
    reduced = CalibrationDatabase(make_cal_db(fringe_visibility=visibility))
    np.testing.assert_allclose(
        reduced.fringe(0, WAVENUMBER), 0.5 * ideal.fringe(0, WAVENUMBER)
    )


def test_spectral_transmission_is_zero_outside_its_table(make_cal_db):
    cal_db = CalibrationDatabase(make_cal_db(transmission=0.8))
    np.testing.assert_allclose(
        cal_db.spectral_transmission([7000.0, 7330.0, 7600.0]), [0.0, 0.8, 0.0]
    )


def test_forward_operator_matches_the_schema_model(make_cal_db):
    etendue = np.full((2, 16), 1.5e-10)
    etendue[1, 3] = 3e-10
    cal_db = CalibrationDatabase(make_cal_db(pixel_etendue=etendue))
    nu = np.array([7328.0, 7329.0])
    bin_width = np.array([0.5, 0.25])
    exposure_time = 0.2

    result = cal_db.forward_operator(1, nu, bin_width, exposure_time)

    photon_energy = 6.62607015e-34 * 299792458.0 * nu * 100
    expected = (
        exposure_time
        / 2.0
        * 3e-10
        * 0.8
        * (1e7 / nu**2)
        * bin_width
        / photon_energy
        * (1 + cal_db.fringe(1, nu)[3])
    )
    np.testing.assert_allclose(result[3], expected, rtol=1e-12)


def test_dataset_is_exposed(make_cal_db):
    ds = make_cal_db()
    assert CalibrationDatabase(ds).ds is ds
    assert isinstance(CalibrationDatabase(ds).ds, xr.Dataset)


def test_bin_width_is_the_trapezoid_rule():
    grid = np.array([0.0, 1.0, 3.0, 4.0])
    np.testing.assert_allclose(bin_width(grid), [0.5, 1.5, 1.5, 0.5])
    np.testing.assert_allclose(
        bin_width(grid) @ (2 * grid + 1), np.trapezoid(2 * grid + 1, grid)
    )


def test_model_wavenumber_spans_the_transmission_table_exactly(make_cal_db):
    cal_db = CalibrationDatabase(make_cal_db())
    table = cal_db.ds["transmission_wavenumber"].to_numpy()
    grid = cal_db.model_wavenumber(resolution_cminv=0.03)
    assert grid[0] == table[0]
    assert grid[-1] == table[-1]
    assert np.max(np.diff(grid)) <= 0.03 + 1e-9
