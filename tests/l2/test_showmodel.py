from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import sasktran2 as sk
import xarray as xr
from skretrieval.core.sasktranformat import SASKTRANRadiance
from skretrieval.retrieval.ancillary import GenericAncillary
from skretrieval.retrieval.measvec import select

from showlib.cal_db import CalibrationDatabase
from showlib.l2.lineshape import SHOWLineShapes
from showlib.l2.showmodel import SHOWForwardModel, SHOWInstrumentModel

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


@pytest.mark.parametrize("order", ["increasing", "shuffled"])
def test_instrument_model_accepts_any_model_order(instrument_model, order):
    rng = np.random.default_rng(4)
    spectra = rng.uniform(1, 2, (len(MODEL_WAVENUMBER), 2))
    wf = rng.uniform(0, 1, (3, len(MODEL_WAVENUMBER), 2))
    expected = instrument_model.model_radiance(_model_radiance(spectra, wf), None)
    index = (
        np.arange(len(MODEL_WAVENUMBER))
        if order == "increasing"
        else rng.permutation(len(MODEL_WAVENUMBER))
    )
    ds = xr.Dataset(
        {
            "radiance": (["wavelength", "los", "stokes"], spectra[index, :, None]),
            "wf_test": (["x", "wavelength", "los", "stokes"], wf[:, index, :, None]),
        },
        coords={"wavelength": 1e7 / MODEL_WAVENUMBER[index]},
    )

    l1 = instrument_model.model_radiance(SASKTRANRadiance.from_sasktran2(ds), None)

    np.testing.assert_allclose(l1.data["radiance"], expected.data["radiance"])
    np.testing.assert_allclose(l1.data["wf_test"], expected.data["wf_test"])


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


def test_measurement_flattening_matches_the_whitener(instrument_model):
    """skretrieval flattens the model output in (wavenumber, los) order, which is
    the order showlib.l2.noise.measurement_whitener assumes."""
    rng = np.random.default_rng(3)
    spectra = rng.uniform(1, 2, (len(MODEL_WAVENUMBER), 2))
    wf = rng.uniform(0, 1, (3, len(MODEL_WAVENUMBER), 2))
    l1 = instrument_model.model_radiance(_model_radiance(spectra, wf), None)
    l1.data["wf"] = l1.data["wf_test"]

    measurement = select({"meas": l1})

    radiance = l1.data["radiance"].transpose("wavenumber", "los").to_numpy()
    num_los = radiance.shape[1]
    np.testing.assert_array_equal(measurement.y, radiance.ravel())
    sample, los = 2, 1
    assert measurement.y[sample * num_los + los] == radiance[sample, los]


def test_forward_model_atmosphere_has_no_thermodynamic_derivatives():
    """They are not state elements, and at the model's size they were most of
    SASKTRAN2's output."""
    altitude = np.arange(0, 60001, 5000.0)
    forward_model = object.__new__(SHOWForwardModel)
    forward_model._model_geometry = {
        "meas": sk.Geometry1D(
            0.6,
            0.0,
            6372e3,
            altitude,
            sk.InterpolationMethod.LinearInterpolation,
            sk.GeometryType.Spherical,
        )
    }
    forward_model._engine_config = sk.Config()
    forward_model._model_wavelength = {"meas": np.array([1364.0, 1365.0])}
    forward_model._state_vector = SimpleNamespace(add_to_atmosphere=lambda _: None)
    forward_model._ancillary = GenericAncillary(
        altitude, 1e5 * np.exp(-altitude / 7000), np.full(len(altitude), 250.0)
    )

    atmo = forward_model._construct_atmosphere()["meas"]

    assert atmo.calculate_derivatives
    assert not atmo.calculate_pressure_derivative
    assert not atmo.calculate_temperature_derivative
    assert not atmo.calculate_specific_humidity_derivative
