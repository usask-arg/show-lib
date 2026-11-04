from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from skretrieval.core.lineshape import Gaussian
from skretrieval.core.sasktranformat import SASKTRANRadiance

from showlib.cal_db import CalibrationDatabase
from showlib.l2.lineshape import SHOWLineShapes
from showlib.l2.showmodel import SHOWBandModel, SHOWForwardModel, SHOWInstrumentModel

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


# ------------------------------------------------------- SHOWInstrumentModel
MODEL_WAVENUMBER = np.linspace(7320.0, 7330.0, 41)
MODEL_SAMPLES = np.array([7322.0, 7325.0, 7328.0])


def _model_radiance(spectra: np.ndarray, wf=None) -> SASKTRANRadiance:
    """SASKTRAN-style radiance in increasing wavelength, spectra (model, los)."""
    order = np.argsort(1e7 / MODEL_WAVENUMBER)
    ds = xr.Dataset(
        {"radiance": (["wavelength", "los", "stokes"], spectra[order, :, None])},
        coords={"wavelength": 1e7 / MODEL_WAVENUMBER[order]},
    )
    if wf is not None:
        ds["wf_test"] = (["x", "wavelength", "los", "stokes"], wf[:, order, :, None])
    return SASKTRANRadiance.from_sasktran2(ds)


@pytest.fixture
def instrument_model() -> SHOWInstrumentModel:
    rng = np.random.default_rng(0)
    matrices = rng.uniform(0, 1, (2, len(MODEL_SAMPLES), len(MODEL_WAVENUMBER)))
    return SHOWInstrumentModel(matrices, MODEL_SAMPLES, MODEL_WAVENUMBER)


def test_instrument_model_applies_each_los_matrix(instrument_model):
    rng = np.random.default_rng(1)
    spectra = rng.uniform(1, 2, (len(MODEL_WAVENUMBER), 2))

    l1 = instrument_model.model_radiance(_model_radiance(spectra), None)

    assert l1.data["radiance"].dims == ("wavenumber", "los")
    assert "los" not in l1.data.coords
    np.testing.assert_array_equal(l1.data.wavenumber, MODEL_SAMPLES)
    for los in range(2):
        np.testing.assert_allclose(
            l1.data["radiance"].isel(los=los),
            instrument_model.matrices[los] @ spectra[:, los],
        )


def test_instrument_model_maps_weighting_functions(instrument_model):
    rng = np.random.default_rng(2)
    spectra = rng.uniform(1, 2, (len(MODEL_WAVENUMBER), 2))
    wf = rng.uniform(0, 1, (3, len(MODEL_WAVENUMBER), 2))

    l1 = instrument_model.model_radiance(_model_radiance(spectra, wf), None)

    assert l1.data["wf_test"].dims == ("wavenumber", "los", "x")
    for los in range(2):
        np.testing.assert_allclose(
            l1.data["wf_test"].isel(los=los),
            instrument_model.matrices[los] @ wf[:, :, los].T,
        )


def test_instrument_model_rejects_a_different_model_grid(instrument_model):
    model = SHOWInstrumentModel(
        instrument_model.matrices, MODEL_SAMPLES, MODEL_WAVENUMBER + 0.1
    )
    spectra = np.ones((len(MODEL_WAVENUMBER), 2))
    with pytest.raises(ValueError, match="model wavenumber"):
        model.model_radiance(_model_radiance(spectra), None)


# ------------------------------------------------------------ SHOWForwardModel
def test_forward_model_hooks_use_the_line_shapes(make_cal_db):
    cal_db = CalibrationDatabase(make_cal_db())
    line_shapes = SHOWLineShapes(cal_db, apodization=None, model_res_cminv=0.05)
    samples = cal_db.sample_wavenumber(0)
    forward_model = object.__new__(SHOWForwardModel)
    forward_model._line_shapes = line_shapes
    forward_model._rows = np.array([1, 0])
    forward_model._get_required_wavelength = lambda: {"meas": 1e7 / samples[[2, 0]]}

    wavelength = forward_model._construct_model_wavelength()["meas"]
    inst = forward_model._construct_inst_model()["meas"]

    np.testing.assert_allclose(wavelength, 1e7 / line_shapes.model_wavenumber[::-1])
    np.testing.assert_allclose(inst.sample_wavenumber, samples[[0, 2]])
    np.testing.assert_allclose(inst.matrices[0], line_shapes.matrix(1)[[0, 2]])
