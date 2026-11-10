from __future__ import annotations

import numpy as np
import pytest

from showlib.processing.l1b_to_l2 import (
    aerosol_extinction_profile,
    stratospheric_aerosol_optical_property,
)

ALTITUDES = np.arange(0.0, 50e3 + 1, 5e3)
PEAK = 4e-7


class _Retrieval:
    """Stands in for skretrieval's Retrieval, which supplies optical properties."""

    def _optical_property(self, name):  # noqa: ARG002
        return stratospheric_aerosol_optical_property()


def _prior_state():
    """A preprocessor-style profile: zero below and above the layer."""
    ext = np.zeros_like(ALTITUDES)
    ext[3:7] = [1e-7, PEAK, 2e-7, 5e-8]
    return ext


def _element(log_space=True, prior_influence=1.0, tikh_factor=0.0, **cfg):
    return aerosol_extinction_profile(
        _Retrieval(),
        "stratospheric_aerosol",
        ALTITUDES,
        {
            "prior_state": _prior_state(),
            "nominal_wavelength": 1350,
            "log_space": log_space,
            "floor_fraction_of_peak": 1e-3,
            "retrieved_quantities": {
                "extinction_per_m": {
                    "prior_influence": prior_influence,
                    "tikh_factor": tikh_factor,
                    "min_value": 0,
                    "max_value": 1e-3,
                }
            },
            "prior": {"extinction_per_m": {"type": "testing"}},
            **cfg,
        },
    )


@pytest.fixture(scope="module")
def element():
    return _element(tikh_factor=0.1, prior_influence=1e-4)


def test_extinction_is_floored_at_a_fraction_of_the_peak(element):
    extinction = element._constituent.extinction_per_m
    np.testing.assert_allclose(extinction.min(), 1e-3 * PEAK)
    np.testing.assert_allclose(extinction[3:7], _prior_state()[3:7])


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_zero_lower_bound_becomes_a_finite_floor_in_log_space():
    """log(0) would be -inf, and skretrieval's scipy minimizer multiplies the bounds
    by a matrix, which turns any infinite bound into NaN for every element."""
    lower = _element().lower_bound()
    assert np.all(np.isfinite(lower))
    np.testing.assert_allclose(lower, np.log(1e-30))


def test_state_and_bounds_are_in_log_space(element):
    np.testing.assert_allclose(
        element.state(), np.log(element._constituent.extinction_per_m)
    )
    assert np.all(element.lower_bound() < element.state())
    np.testing.assert_allclose(element.upper_bound(), np.log(1e-3))


def test_prior_agrees_with_the_starting_state(element):
    np.testing.assert_allclose(element.apriori_state(), element.state())
    residual = element.prior_precision_factor() @ (
        element.state() - element.apriori_state()
    )
    np.testing.assert_allclose(residual, 0.0, atol=1e-12)


def test_prior_penalizes_fractional_changes_equally_at_any_magnitude():
    diagonal = _element(prior_influence=1.0, tikh_factor=0.0)
    weak_level, strong_level = 0, 4
    norms = []
    for level in (weak_level, strong_level):
        delta = np.zeros_like(diagonal.state())
        delta[level] = np.log(2.0)  # double the extinction
        norms.append(np.linalg.norm(diagonal.prior_precision_factor() @ delta))
    np.testing.assert_allclose(norms[0], norms[1])


def test_linear_space_is_unchanged():
    linear = _element(log_space=False, tikh_factor=0.1, prior_influence=1e-4)
    np.testing.assert_allclose(linear.state(), linear._constituent.extinction_per_m)
    np.testing.assert_allclose(linear.apriori_state(), linear.state())
