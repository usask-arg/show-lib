from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from skretrieval.core.radianceformat import RadianceGridded

from showlib.cal_db import CalibrationDatabase
from showlib.l2.lineshape import SHOWLineShapes
from showlib.l2.noise import noise_correlation_applies, whitened_measurement_vector
from showlib.processing.l1a_to_l1b import inverse_operator

SAMPLE_INDICES = (-6, -5, -4, -3, -2, -1)


def _line_shapes(make_cal_db, apodization=10.0, **kwargs) -> SHOWLineShapes:
    cal_db = CalibrationDatabase(make_cal_db(sample_indices=SAMPLE_INDICES, **kwargs))
    return SHOWLineShapes(cal_db, apodization=apodization, model_res_cminv=0.05)


def test_noise_correlation_matches_monte_carlo(make_cal_db):
    line_shapes = _line_shapes(make_cal_db)
    operator = inverse_operator(line_shapes.cal_db, 0, apodization=10.0)
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(40_000, operator.shape[1])) @ operator.T

    np.testing.assert_allclose(
        line_shapes.noise_correlation(0), np.corrcoef(samples.T), atol=0.02
    )


def test_kaiser_correlates_neighbouring_samples(make_cal_db):
    correlation = _line_shapes(make_cal_db).noise_correlation(0)
    np.testing.assert_allclose(np.diag(correlation), 1.0)
    np.testing.assert_allclose(correlation, correlation.T)
    assert np.mean(np.diag(correlation, 1)) > 0.5


def test_rectangular_window_on_an_ideal_grid_is_uncorrelated(make_cal_db):
    correlation = _line_shapes(make_cal_db, apodization=None).noise_correlation(0)
    np.testing.assert_allclose(correlation, np.eye(len(SAMPLE_INDICES)), atol=1e-10)


def test_noise_correlation_for_selected_samples(make_cal_db):
    line_shapes = _line_shapes(make_cal_db)
    samples = line_shapes.cal_db.sample_wavenumber(0)[[1, 3, 4]]
    full = line_shapes.noise_correlation(0)
    np.testing.assert_allclose(
        line_shapes.noise_correlation(0, samples), full[np.ix_([1, 3, 4], [1, 3, 4])]
    )


# ------------------------------------------------------- whitened measurement
ROWS = np.array([1, 0])


def _observation(line_shapes, radiance=None, wf=None) -> RadianceGridded:
    samples = line_shapes.cal_db.sample_wavenumber(0)[1:]
    num_samples, num_los = len(samples), len(ROWS)
    rng = np.random.default_rng(1)
    if radiance is None:
        radiance = rng.uniform(1, 2, (num_samples, num_los))
    if wf is None:
        wf = rng.normal(size=(num_samples, num_los, 2))
    return RadianceGridded(
        xr.Dataset(
            {
                "radiance": (["wavenumber", "los"], radiance),
                "radiance_noise": (
                    ["wavenumber", "los"],
                    np.linspace(0.01, 0.03, num_samples * num_los).reshape(
                        num_samples, num_los
                    ),
                ),
                "wf": (["wavenumber", "los", "x"], wf),
            },
            coords={"wavenumber": samples},
        )
    )


def _inverse_covariance(line_shapes, observation: RadianceGridded) -> np.ndarray:
    """Sy^-1 in the (wavenumber, los) flattened order skretrieval uses."""
    sigma = observation.data["radiance_noise"].to_numpy()
    samples = observation.data["wavenumber"].to_numpy()
    num_samples, num_los = sigma.shape
    covariance = np.zeros((num_samples * num_los,) * 2)
    for los, row in enumerate(ROWS):
        correlation = line_shapes.noise_correlation(row, samples)
        index = np.arange(num_samples) * num_los + los
        covariance[np.ix_(index, index)] = (
            sigma[:, los, None] * correlation * sigma[None, :, los]
        )
    return np.linalg.inv(covariance)


def test_whitening_preserves_the_correlated_chi_square(make_cal_db):
    line_shapes = _line_shapes(
        make_cal_db, littrow_edges=np.array([7330.0, 7330.2, 7330.5])
    )
    observation = _observation(line_shapes)
    measurement_vector = whitened_measurement_vector(
        observation.data, line_shapes, ROWS
    )

    measurement = measurement_vector.fn({"meas": observation}, None)

    inverse = _inverse_covariance(line_shapes, observation)
    y = observation.data["radiance"].to_numpy().flatten()
    jacobian = observation.data["wf"].to_numpy().reshape(len(y), -1)
    np.testing.assert_allclose(measurement.y @ measurement.y, y @ inverse @ y)
    np.testing.assert_allclose(
        measurement.K.T @ measurement.K, jacobian.T @ inverse @ jacobian
    )
    np.testing.assert_allclose(
        measurement.K.T @ measurement.y, jacobian.T @ inverse @ y
    )
    np.testing.assert_allclose(measurement.Sy.toarray(), np.eye(len(y)))


def test_whitening_without_correlation_divides_by_the_noise(make_cal_db):
    line_shapes = _line_shapes(make_cal_db, apodization=None)
    observation = _observation(line_shapes)
    measurement_vector = whitened_measurement_vector(
        observation.data, line_shapes, ROWS
    )

    measurement = measurement_vector.fn({"meas": observation}, None)

    expected = (
        observation.data["radiance"] / observation.data["radiance_noise"]
    ).to_numpy()
    np.testing.assert_allclose(measurement.y, expected.flatten(), atol=1e-9)


def test_whitening_rejects_a_different_measurement_size(make_cal_db):
    line_shapes = _line_shapes(make_cal_db)
    observation = _observation(line_shapes)
    measurement_vector = whitened_measurement_vector(
        observation.data, line_shapes, ROWS
    )
    smaller = RadianceGridded(observation.data.isel(wavenumber=slice(1, None)))

    with pytest.raises(ValueError, match="measurement"):
        measurement_vector.fn({"meas": smaller}, None)


def test_whitening_rejects_non_positive_noise(make_cal_db):
    line_shapes = _line_shapes(make_cal_db)
    observation = _observation(line_shapes)
    observation.data["radiance_noise"][0, 1] = 0.0

    with pytest.raises(ValueError, match="radiance_noise"):
        whitened_measurement_vector(observation.data, line_shapes, ROWS)


@pytest.mark.parametrize(
    ("noise_model", "expected"),
    [
        ("detector", True),
        ("detector+fractional", False),
        ("fractional", False),
        ("none", False),
    ],
)
def test_correlation_applies_only_to_pure_detector_noise(noise_model, expected):
    """The fractional term is uncorrelated between samples, and L1B stores only the
    total, so the B-derived correlation describes the noise only without it."""
    assert noise_correlation_applies({"radiance_noise_model": noise_model}) is expected
