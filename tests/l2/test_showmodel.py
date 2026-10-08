from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from skretrieval.core.lineshape import Gaussian
from skretrieval.core.sasktranformat import SASKTRANRadiance

from showlib.l2.showmodel import SHOWBandModel

HIRES_WAVENUMBER = np.arange(7300.0, 7330.0, 0.01)
SAMPLE_WAVENUMBER = np.array([7310.0, 7315.0, 7320.0])
NUM_LOS = 2


def _sasktran_radiance(spectrum: np.ndarray, wf=None) -> SASKTRANRadiance:
    rad = spectrum[:, np.newaxis] * (1 + np.arange(NUM_LOS))[np.newaxis, :]
    ds = xr.Dataset(
        {"radiance": (["wavelength", "los", "stokes"], rad[:, :, np.newaxis])},
        coords={"wavelength": 1e7 / HIRES_WAVENUMBER},
    )
    if wf is not None:
        ds["wf_test"] = (
            ["x", "wavelength", "los", "stokes"],
            wf[:, :, np.newaxis, np.newaxis] * rad[np.newaxis, :, :, np.newaxis],
        )
    return SASKTRANRadiance.from_sasktran2(ds)


@pytest.fixture
def model() -> SHOWBandModel:
    return SHOWBandModel(SAMPLE_WAVENUMBER, Gaussian(fwhm=0.3))


def test_constant_radiance_is_preserved(model):
    l1 = model.model_radiance(_sasktran_radiance(np.full(len(HIRES_WAVENUMBER), 3.0)))

    assert l1.data["radiance"].dims == ("wavenumber", "los")
    np.testing.assert_array_equal(l1.data.wavenumber, SAMPLE_WAVENUMBER)
    np.testing.assert_allclose(l1.data["radiance"].isel(los=0), 3.0, rtol=1e-10)
    np.testing.assert_allclose(l1.data["radiance"].isel(los=1), 6.0, rtol=1e-10)


def test_linear_radiance_samples_at_centre(model):
    spectrum = 1 + 0.01 * (HIRES_WAVENUMBER - 7300)

    l1 = model.model_radiance(_sasktran_radiance(spectrum))

    expected = 1 + 0.01 * (SAMPLE_WAVENUMBER - 7300)
    np.testing.assert_allclose(l1.data["radiance"].isel(los=0), expected, rtol=1e-8)


def test_weighting_functions_are_convolved(model):
    spectrum = np.full(len(HIRES_WAVENUMBER), 2.0)
    wf = np.ones((4, len(HIRES_WAVENUMBER))) * np.arange(1, 5)[:, np.newaxis]

    l1 = model.model_radiance(_sasktran_radiance(spectrum, wf=wf))

    assert l1.data["wf_test"].dims == ("wavenumber", "los", "x")
    np.testing.assert_allclose(
        l1.data["wf_test"].isel(los=0),
        2.0 * np.arange(1, 5)[np.newaxis, :] * np.ones((3, 1)),
    )


def test_interpolator_is_cached(model):
    radiance = _sasktran_radiance(np.ones(len(HIRES_WAVENUMBER)))

    model.model_radiance(radiance)
    interpolator = model._interpolator
    model.model_radiance(radiance)

    assert model._interpolator is interpolator
    assert model._interpolator.shape == (len(SAMPLE_WAVENUMBER), len(HIRES_WAVENUMBER))
