from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from skretrieval.core.sasktranformat import SASKTRANRadiance
from skretrieval.retrieval.statevector import StateVectorElement
from skretrieval.retrieval.statevector.altitude import AltitudeNativeStateVector
from skretrieval.retrieval.statevector.shifts import WavenumberShift
from skretrieval.retrieval.statevector.spline import MultiplicativeSplineOne

from showlib.l2.retrieval import SHOWRetrieval, SHOWStateVector
from showlib.l2.showmodel import SHOWInstrumentModel

NUM_LOS = 3
NUM_ALT = 4
WAVELENGTH = np.linspace(1362.0, 1368.0, 25)


class _Constituent(StateVectorElement):
    """Takes its weighting functions from SASKTRAN2's wf_fake, like a constituent."""

    def __init__(self, enabled=True):
        super().__init__(enabled)
        self._x = np.ones(NUM_ALT)

    def state(self):
        return self._x

    def name(self):
        return "fake"

    def update_state(self, x):
        self._x = x

    def propagate_wf(self, radiance):
        return radiance["wf_fake"].rename({"altitude": "x"}) * 2

    def modify_input_radiance(self, radiance):
        return radiance


def _elements(constituent_enabled=True) -> dict:
    shift = WavenumberShift(NUM_LOS, tikh_factor=1.0, numerical_delta=1e-3)
    shift.update_state(np.array([0.01, -0.02, 0.005]))
    spline = MultiplicativeSplineOne(
        WAVELENGTH[1], WAVELENGTH[-2], num_wv=2, s=1, order=1
    )
    spline.update_state(np.array([0.9, 1.2]))
    return {
        "fake": _Constituent(constituent_enabled),
        "shift": shift,
        "spline": spline,
    }


@pytest.fixture
def radiance() -> xr.Dataset:
    rng = np.random.default_rng(0)
    shape = (len(WAVELENGTH), NUM_LOS, 1)
    return xr.Dataset(
        {
            "radiance": (["wavelength", "los", "stokes"], rng.uniform(1, 2, shape)),
            "wf_fake": (
                ["altitude", "wavelength", "los", "stokes"],
                rng.uniform(0, 1, (NUM_ALT, *shape)),
            ),
            # SASKTRAN2 output that no state element uses
            "wf_temperature_k": (
                ["altitude", "wavelength", "los", "stokes"],
                rng.uniform(0, 1, (NUM_ALT, *shape)),
            ),
        },
        coords={"wavelength": WAVELENGTH},
    )


def _expand(result: xr.Dataset) -> np.ndarray:
    """SHOWStateVector's weighting functions as the dense (x, wavelength, los,
    stokes) array skretrieval gives."""
    dims = ("x", "wavelength", "los", "stokes")
    dense = result["wf"].transpose(*dims)
    num_x = dense.sizes["x"]
    if "wf_per_los" in result:
        num_x += result.sizes["x_per_los"] * NUM_LOS
    wf = np.zeros((num_x, *dense.shape[1:]))
    wf[dense["x"].to_numpy()] = dense.to_numpy()
    if "wf_per_los" in result:
        per_los = result["wf_per_los"].transpose("x_per_los", *dims[1:])
        for start, element_wf in zip(
            result["x_per_los"].to_numpy(), per_los.to_numpy(), strict=True
        ):
            for los in range(NUM_LOS):
                wf[start + los, :, los] = element_wf[:, los]
    return wf


def _post_process(state_vector_class, radiance, elements):
    return state_vector_class(
        np.arange(NUM_ALT, dtype=float), **elements
    ).post_process_sk2_radiances(radiance.copy(deep=True))


@pytest.mark.parametrize("constituent_enabled", [True, False])
@pytest.mark.parametrize("shift_first", [False, True])
def test_post_processing_matches_skretrieval(
    radiance, constituent_enabled, shift_first
):
    def elements():
        elements = _elements(constituent_enabled)
        if shift_first:
            elements = {"shift": elements.pop("shift"), **elements}
        return elements

    expected = _post_process(AltitudeNativeStateVector, radiance, elements())
    result = _post_process(SHOWStateVector, radiance, elements())

    assert sorted(result.data_vars) == ["radiance", "wf", "wf_per_los"]
    np.testing.assert_allclose(result["radiance"], expected["radiance"])
    # The shift is kept as one derivative per line of sight, not num_los columns
    assert result.sizes["x_per_los"] == 1
    np.testing.assert_allclose(
        _expand(result),
        expected["wf"].transpose("x", "wavelength", "los", "stokes"),
        rtol=1e-12,
        atol=1e-12,
    )


def test_post_processing_without_a_shift_is_dense(radiance):
    elements = _elements()
    del elements["shift"]

    expected = _post_process(AltitudeNativeStateVector, radiance, elements)
    result = _post_process(SHOWStateVector, radiance, elements)

    assert "wf_per_los" not in result
    np.testing.assert_allclose(
        result["wf"].transpose("x", "wavelength", "los", "stokes"),
        expected["wf"].transpose("x", "wavelength", "los", "stokes"),
    )


def test_instrument_model_matches_dense_weighting_functions(radiance):
    """On the sample grid the compact shift gives skretrieval's weighting
    functions, in state-vector order."""
    rng = np.random.default_rng(1)
    model_wavenumber = np.sort(1e7 / WAVELENGTH)
    model = SHOWInstrumentModel(
        rng.uniform(0, 1, (NUM_LOS, 5, len(WAVELENGTH))),
        np.linspace(model_wavenumber[2], model_wavenumber[-3], 5),
        model_wavenumber,
    )
    expected = _post_process(AltitudeNativeStateVector, radiance, _elements())
    result = _post_process(SHOWStateVector, radiance, _elements())

    mapped = model.model_radiance(SASKTRANRadiance.from_sasktran2(result))
    dense = model.model_radiance(SASKTRANRadiance.from_sasktran2(expected))

    assert "wf_per_los" not in mapped.data
    assert mapped.data["wf"].dims == ("wavenumber", "los", "x")
    np.testing.assert_allclose(
        mapped.data["wf"], dense.data["wf"], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(mapped.data["radiance"], dense.data["radiance"])


def test_show_retrieval_uses_the_show_state_vector():
    retrieval = object.__new__(SHOWRetrieval)
    retrieval._state_fns = SHOWRetrieval._state_fns
    retrieval._state_kwargs = {
        "altitude_grid": np.arange(NUM_ALT, dtype=float),
        "shifts": {"wavenumber": {"type": "wavenumber_shift", "num_los": NUM_LOS}},
    }

    state_vector = retrieval._construct_state_vector()

    assert isinstance(state_vector, SHOWStateVector)
    assert list(state_vector.sv) == ["wavenumber"]
