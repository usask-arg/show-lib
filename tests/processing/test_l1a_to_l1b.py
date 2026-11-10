from __future__ import annotations

import numpy as np
import pytest

from showlib.cal_db import CalibrationDatabase, bin_width
from showlib.l1a.data import L1ADataSet, L1AImage
from showlib.l2.lineshape import instrument_line_shape
from showlib.processing.l1a_to_l1b import inverse_operator, process_l1a_to_l1b
from showlib.schemas import SchemaValidationError, validate

EXPOSURE_TIMES = (0.2, 0.5)
TIMES = ("2026-11-03T12:00:00", "2026-11-03T12:00:01")


def _radiance(cal_db: CalibrationDatabase) -> np.ndarray:
    """A distinct test spectrum per (row, sample): shape (los, spectral_sample)."""
    num_samples = cal_db.ds.sizes["spectral_sample"]
    rows = np.arange(cal_db.num_rows)[:, np.newaxis]
    return 1e-3 * (1 + rows + 0.5 * np.arange(num_samples)[np.newaxis, :])


def _l1a(cal_db: CalibrationDatabase, radiance=None, saturated=None) -> L1ADataSet:
    """Noise-free L1A generated with the forward operator at the sample grid."""
    radiance = _radiance(cal_db) if radiance is None else radiance
    num_rows = cal_db.num_rows
    images = []
    for time, exposure_time in zip(TIMES, EXPOSURE_TIMES, strict=True):
        image = np.stack(
            [
                cal_db.forward_operator(
                    row,
                    cal_db.sample_wavenumber(row),
                    np.gradient(cal_db.sample_wavenumber(row)),
                    exposure_time,
                )
                @ radiance[row]
                for row in range(num_rows)
            ]
        )
        images.append(
            L1AImage(
                image=image,
                time=np.datetime64(time),
                tangent_altitude=np.linspace(10e3, 20e3, num_rows),
                tangent_latitude=np.full(num_rows, 52.0),
                tangent_longitude=np.full(num_rows, -106.0),
                spacecraft_latitude=50.0,
                spacecraft_longitude=-106.0,
                spacecraft_altitude=500e3,
                solar_zenith_angle=np.full(num_rows, 60.0),
                relative_solar_azimuth_angle=np.full(num_rows, 30.0),
                los_azimuth_angle=np.full(num_rows, 90.0),
                exposure_time=exposure_time,
                num_averaged_exposures=4,
                saturated=saturated,
            )
        )
    return L1ADataSet.from_images(images)


def _general_cal_db(make_cal_db) -> CalibrationDatabase:
    """Every generalization at once: non-uniform OPD, tilt, smile, phase,
    dispersion, and per-pixel etendue and visibility."""
    num_rows, num_columns = 3, 24
    rng = np.random.default_rng(1)
    edges = (np.arange(num_columns + 1) - num_columns / 2) * 0.1
    edges = edges + 0.01 * edges**2
    opd = edges + np.array([[0.0], [0.01], [0.025], [0.03]])
    ds = make_cal_db(
        num_rows=num_rows,
        num_columns=num_columns,
        opd=opd,
        littrow_edges=np.array([7330.0, 7330.1, 7330.15, 7330.3]),
        fringe_phase=rng.uniform(-0.3, 0.3, opd.shape),
        fringe_visibility=rng.uniform(0.8, 1.0, (num_rows, num_columns)),
        pixel_etendue=rng.uniform(1e-10, 2e-10, (num_rows, num_columns)),
        sample_indices=(-6, -5, -4, -3, -2),
    )
    grid = np.linspace(7320.0, 7340.0, 41)
    offset = grid - 7330.0
    ds = ds.assign(
        fringe_frequency_scale=("dispersion_wavenumber", 1 + 2e-4 * offset**2),
        dispersion_phase=("dispersion_wavenumber", 0.01 * offset**2),
        dispersion_reference_wavenumber=((), 7330.0),
    ).assign_coords(dispersion_wavenumber=grid)
    for name, units in [
        ("fringe_frequency_scale", "1"),
        ("dispersion_phase", "rad"),
        ("dispersion_reference_wavenumber", "cm-1"),
        ("dispersion_wavenumber", "cm-1"),
    ]:
        ds[name].attrs["units"] = units
    return CalibrationDatabase(ds)


def test_output_is_a_valid_l1b_granule(make_cal_db):
    cal_db = CalibrationDatabase(make_cal_db())
    l1a = _l1a(cal_db)

    l1b = process_l1a_to_l1b(l1a, cal_db)

    assert validate(l1b.ds, "l1b") == []
    np.testing.assert_array_equal(l1b.ds["wavenumber"], cal_db.ds["sample_wavenumber"])
    np.testing.assert_array_equal(
        l1b.ds["tangent_altitude"], l1a.ds["tangent_altitude"]
    )
    np.testing.assert_array_equal(l1b.ds["time"], l1a.ds["time"])
    np.testing.assert_array_equal(l1b.ds["los"], np.arange(cal_db.num_rows))


def test_kaiser_apodization_is_the_default(make_cal_db):
    cal_db = CalibrationDatabase(make_cal_db())
    l1a = _l1a(cal_db)

    default = process_l1a_to_l1b(l1a, cal_db)
    explicit = process_l1a_to_l1b(l1a, cal_db, apodization=10.0)
    rectangular = process_l1a_to_l1b(l1a, cal_db, apodization=None)

    np.testing.assert_allclose(default.ds["radiance"], explicit.ds["radiance"])
    assert not np.allclose(default.ds["radiance"], rectangular.ds["radiance"])
    assert default.ds.attrs["apodization_window"] == "kaiser"
    assert default.ds.attrs["apodization_beta"] == 10.0
    assert rectangular.ds.attrs["apodization_window"] == "rectangular"
    assert "apodization_beta" not in rectangular.ds.attrs


@pytest.mark.parametrize("apodization", [-1.0, np.inf, np.nan])
def test_invalid_apodization_is_rejected(make_cal_db, apodization):
    cal_db = CalibrationDatabase(make_cal_db())
    with pytest.raises(ValueError, match="apodization"):
        process_l1a_to_l1b(_l1a(cal_db), cal_db, apodization=apodization)


def test_detector_noise_is_propagated(make_cal_db):
    cal_db = CalibrationDatabase(make_cal_db())
    l1a = _l1a(cal_db)

    l1b = process_l1a_to_l1b(l1a, cal_db, apodization=None)

    gain, read_noise, num_exposures = 2.0, 10.0, 4
    time_index, row = 1, 1
    counts = l1a.ds["image"].isel(time=time_index, los=row).to_numpy()
    exposure_time = EXPOSURE_TIMES[time_index]
    count_variance = (counts * gain + read_noise**2 + gain**2 / 12) / (
        num_exposures * gain**2
    )
    operator = inverse_operator(cal_db, row, apodization=None)
    expected = np.sqrt(operator**2 @ (count_variance / exposure_time**2))

    np.testing.assert_allclose(
        l1b.ds["radiance_noise"].isel(time=time_index, los=row), expected
    )
    assert l1b.ds.attrs["radiance_noise_model"] == "detector"


def test_noise_fraction_adds_in_quadrature(make_cal_db):
    cal_db = CalibrationDatabase(make_cal_db())
    l1a = _l1a(cal_db)

    detector = process_l1a_to_l1b(l1a, cal_db)
    combined = process_l1a_to_l1b(l1a, cal_db, noise_fraction=0.01)

    np.testing.assert_allclose(
        combined.ds["radiance_noise"],
        np.hypot(detector.ds["radiance_noise"], 0.01 * np.abs(detector.ds["radiance"])),
    )
    assert combined.ds.attrs["radiance_noise_model"] == "detector+fractional"


def test_invalid_noise_fraction_is_rejected(make_cal_db):
    cal_db = CalibrationDatabase(make_cal_db())
    with pytest.raises(ValueError, match="noise_fraction"):
        process_l1a_to_l1b(_l1a(cal_db), cal_db, noise_fraction=-0.1)


def test_saturated_pixels_set_the_row_saturated_fraction(make_cal_db):
    cal_db = CalibrationDatabase(make_cal_db())
    saturated = np.zeros((cal_db.num_rows, cal_db.num_columns), dtype=bool)
    saturated[0, 3] = True

    l1b = process_l1a_to_l1b(_l1a(cal_db, saturated=saturated), cal_db)

    np.testing.assert_allclose(
        l1b.ds["saturated_fraction"].isel(time=0), [1 / cal_db.num_columns, 0.0]
    )
    assert np.all(np.isfinite(l1b.ds["radiance_noise"]))


def test_detector_must_match_the_calibration(make_cal_db):
    l1a = _l1a(CalibrationDatabase(make_cal_db(num_rows=3)))
    with pytest.raises(SchemaValidationError, match="detector-matches-calibration"):
        process_l1a_to_l1b(l1a, CalibrationDatabase(make_cal_db(num_rows=2)))


# ---------------------------------------------------- continuum radiometry
def _offset_zpd_cal_db(make_cal_db):
    num_columns, shift = 64, 16
    edges = (np.arange(num_columns + 1) - num_columns / 2 + shift) * 0.1
    return CalibrationDatabase(
        make_cal_db(
            num_columns=num_columns,
            opd=np.broadcast_to(edges, (3, num_columns + 1)).copy(),
            sample_indices=(-12, -10, -8, -6, -4),
        )
    )


CONTINUUM_CAL_DBS = {
    "ideal": lambda make: CalibrationDatabase(make()),
    "every_second_sample": lambda make: CalibrationDatabase(
        make(num_columns=32, sample_indices=(-10, -8, -6, -4))
    ),
    "offset_zpd": _offset_zpd_cal_db,
    "general": _general_cal_db,
}


def _continuum_l1b(cal_db, spectrum_fn, apodization):
    """L1B from L1A simulated on a fine grid for a spectrum defined everywhere."""
    grid = cal_db.model_wavenumber(resolution_cminv=0.002)
    spectrum = spectrum_fn(grid)
    image = np.stack(
        [
            cal_db.forward_operator(row, grid, bin_width(grid), 0.3) @ spectrum
            for row in range(cal_db.num_rows)
        ]
    )
    num_rows = cal_db.num_rows
    l1a = L1ADataSet.from_images(
        [
            L1AImage(
                image=image,
                time=np.datetime64(TIMES[0]),
                tangent_altitude=np.linspace(10e3, 20e3, num_rows),
                tangent_latitude=np.full(num_rows, 52.0),
                tangent_longitude=np.full(num_rows, -106.0),
                spacecraft_latitude=50.0,
                spacecraft_longitude=-106.0,
                spacecraft_altitude=500e3,
                solar_zenith_angle=np.full(num_rows, 60.0),
                relative_solar_azimuth_angle=np.full(num_rows, 30.0),
                los_azimuth_angle=np.full(num_rows, 90.0),
                exposure_time=0.3,
            )
        ]
    )
    return process_l1a_to_l1b(l1a, cal_db, apodization=apodization)


@pytest.mark.parametrize("apodization", [None, 10.0])
@pytest.mark.parametrize("name", CONTINUUM_CAL_DBS)
def test_flat_continuum_is_recovered(make_cal_db, name, apodization):
    cal_db = CONTINUUM_CAL_DBS[name](make_cal_db)

    l1b = _continuum_l1b(cal_db, lambda nu: np.full(nu.shape, 2e-3), apodization)

    # Set by the default normalization grid (20 samples per fringe period)
    np.testing.assert_allclose(l1b.ds["radiance"], 2e-3, rtol=1e-3)


@pytest.mark.parametrize("apodization", [None, 10.0])
def test_sloped_continuum_is_recovered_at_interior_samples(make_cal_db, apodization):
    """Transmission stops below Littrow: light at the mirror wavenumbers would
    alias onto the same fringes, and with a sloped spectrum bias the result."""
    cal_db = CalibrationDatabase(
        make_cal_db(
            num_columns=64,
            sample_indices=range(-20, -4),
            transmission_range=(7325.0, 7329.6),
        )
    )

    def slope(nu):
        return 2e-3 * (1 + 0.01 * (nu - 7330.0))

    l1b = _continuum_l1b(cal_db, slope, apodization)

    interior = slice(3, -3)
    samples = cal_db.sample_wavenumber(0)[interior]
    np.testing.assert_allclose(
        l1b.ds["radiance"].isel(time=0, los=0, spectral_sample=interior),
        slope(samples),
        rtol=1e-3,
    )


def test_apodization_broadens_the_line_shape(make_cal_db):
    cal_db = CalibrationDatabase(
        make_cal_db(num_columns=64, sample_indices=range(-20, -4))
    )
    grid = cal_db.model_wavenumber(resolution_cminv=0.005)
    sample = 8

    peaks = [
        np.max(instrument_line_shape(cal_db, 0, grid, apodization)[sample])
        for apodization in (None, 10.0)
    ]

    # Both have unit area, so the broader (apodized) line shape has the lower peak
    assert peaks[1] < 0.7 * peaks[0]


def test_a_single_sample_row_recovers_a_flat_continuum(make_cal_db):
    cal_db = CalibrationDatabase(make_cal_db(sample_indices=(-2,)))
    l1b = _continuum_l1b(cal_db, lambda nu: np.full(nu.shape, 2e-3), apodization=None)
    np.testing.assert_allclose(l1b.ds["radiance"], 2e-3, rtol=1e-3)
