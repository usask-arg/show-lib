from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from showlib.l2.shifts import AltitudeShift, BandShifts
from showlib.l2.spline import (
    AddFactors,
    MultiplicativeSpline,
    MultiplicativeSplineOne,
    ScaleFactors,
    ScaleFactorsPoly,
)

NUM_LOS = 3

# (factory, finite difference step). Every element starts at a state that leaves the radiance unchanged,
# and the splines use s=0 so they are linear in their knot values.
ELEMENTS = {
    "multiplicative_spline": (
        lambda: MultiplicativeSpline(NUM_LOS, 1363.0, 1367.0, num_wv=6, s=0),
        1e-3,
    ),
    "multiplicative_spline_one": (
        lambda: MultiplicativeSplineOne(1363.0, 1367.0, num_wv=6, s=0),
        1e-2,
    ),
    "scale_factors": (lambda: ScaleFactors(NUM_LOS), 1e-3),
    "scale_factors_poly": (lambda: ScaleFactorsPoly(NUM_LOS, order=2), 1e-3),
    "add_factors": (lambda: AddFactors(NUM_LOS, order=0), 1e-3),
    "add_factors_poly": (lambda: AddFactors(NUM_LOS, order=1), 1e-3),
    "band_shifts": (lambda: BandShifts(NUM_LOS), 1e-4),
    "altitude_shift": (AltitudeShift, 1e-4),
}


@pytest.fixture
def radiance() -> xr.Dataset:
    wavelength = np.linspace(1362.0, 1368.0, 121)
    spectrum = 1 + 0.2 * np.sin(3 * wavelength)
    rad = spectrum[:, np.newaxis] * (1 + 0.1 * np.arange(NUM_LOS))[np.newaxis, :]

    return xr.Dataset(
        {"radiance": (["wavelength", "los", "stokes"], rad[:, :, np.newaxis])},
        coords={
            "wavelength": wavelength,
            "angle": ("los", 0.1 * np.arange(1, NUM_LOS + 1)),
        },
    )


def _forward(element, radiance: xr.Dataset, x: np.ndarray) -> np.ndarray:
    element.update_state(x.copy())
    return element.modify_input_radiance(radiance.copy(deep=True))[
        "radiance"
    ].to_numpy()


def _numerical_jacobian(element, radiance: xr.Dataset, step: float) -> np.ndarray:
    x0 = element.state().astype(float)
    base = _forward(element, radiance, x0)

    columns = []
    for j in range(len(x0)):
        x = x0.copy()
        x[j] += step
        columns.append((_forward(element, radiance, x) - base) / step)

    element.update_state(x0)
    return np.stack(columns)


@pytest.mark.parametrize("name", ELEMENTS)
def test_state_vector_interface(name):
    element = ELEMENTS[name][0]()

    n = len(element.state())
    assert n > 0
    assert element.lower_bound().shape == (n,)
    assert element.upper_bound().shape == (n,)
    assert element.apriori_state().shape == (n,)
    assert np.all(element.lower_bound() <= element.apriori_state())
    assert np.all(element.apriori_state() <= element.upper_bound())

    inv_cov = element.inverse_apriori_covariance()
    assert np.shape(np.atleast_2d(inv_cov))[-1] == n

    new_state = element.apriori_state() + 0.01
    element.update_state(new_state)
    np.testing.assert_allclose(element.state(), new_state)


def test_shift_names_are_distinct():
    assert BandShifts(NUM_LOS).name() != AltitudeShift().name()


@pytest.mark.parametrize("name", ELEMENTS)
def test_apriori_state_leaves_radiance_unchanged(name, radiance):
    element = ELEMENTS[name][0]()

    modified = _forward(element, radiance, element.apriori_state())

    np.testing.assert_allclose(modified, radiance["radiance"].to_numpy(), rtol=1e-12)


@pytest.mark.parametrize("name", ELEMENTS)
def test_jacobian_matches_finite_difference(name, radiance):
    factory, step = ELEMENTS[name]
    element = factory()

    analytic = element.propagate_wf(radiance.copy(deep=True))
    numerical = _numerical_jacobian(element, radiance, step)

    assert analytic.dims == ("x", "wavelength", "los", "stokes")
    assert analytic.shape == numerical.shape
    scale = np.abs(numerical).max()
    np.testing.assert_allclose(analytic.to_numpy(), numerical, atol=1e-6 * scale)


def test_spline_is_one_outside_its_range(radiance):
    element = MultiplicativeSpline(NUM_LOS, 1363.0, 1367.0, num_wv=6, s=0)
    element.update_state(np.full_like(element.state(), 1.2))

    modified = element.modify_input_radiance(radiance.copy(deep=True))
    ratio = (modified["radiance"] / radiance["radiance"]).isel(stokes=0)

    inside = (radiance.wavelength > 1363.0) & (radiance.wavelength < 1367.0)
    np.testing.assert_allclose(ratio.where(inside, drop=True), 1.2)
    np.testing.assert_allclose(ratio.where(~inside, drop=True), 1.0)


def test_scale_factors_poly_applies_polynomial(radiance):
    element = ScaleFactorsPoly(NUM_LOS, order=1)
    x = element.apriori_state().reshape(NUM_LOS, 2)
    x[:, 1] = 0.1
    element.update_state(x.flatten())

    modified = element.modify_input_radiance(radiance.copy(deep=True))

    w = radiance.wavelength - radiance.wavelength[0]
    xr.testing.assert_allclose(
        modified["radiance"], radiance["radiance"] * (1 + 0.1 * w)
    )


def test_band_shift_moves_spectrum(radiance):
    element = BandShifts(NUM_LOS)
    element.update_state(np.array([0.0, 0.05, 0.0]))

    modified = element.modify_input_radiance(radiance.copy(deep=True))

    shifted = radiance.isel(los=1, stokes=0)["radiance"].interp(
        wavelength=radiance.wavelength + 0.05
    )
    inner = slice(0, -10)
    np.testing.assert_allclose(
        modified["radiance"].isel(los=1, stokes=0)[inner], shifted[inner]
    )
    xr.testing.assert_equal(
        modified["radiance"].isel(los=[0, 2]), radiance["radiance"].isel(los=[0, 2])
    )
