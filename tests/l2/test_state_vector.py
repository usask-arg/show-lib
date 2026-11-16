from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from skretrieval.retrieval.statevector import StateVectorElement
from skretrieval.retrieval.statevector.altitude import AltitudeNativeStateVector
from skretrieval.retrieval.statevector.shifts import WavenumberShift
from skretrieval.retrieval.statevector.spline import MultiplicativeSplineOne

from showlib.l2.retrieval import SHOWRetrieval, SHOWStateVector

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


@pytest.mark.parametrize("constituent_enabled", [True, False])
def test_post_processing_matches_skretrieval(radiance, constituent_enabled):
    altitude = np.arange(NUM_ALT, dtype=float)
    expected = AltitudeNativeStateVector(
        altitude, **_elements(constituent_enabled)
    ).post_process_sk2_radiances(radiance.copy(deep=True))
    result = SHOWStateVector(
        altitude, **_elements(constituent_enabled)
    ).post_process_sk2_radiances(radiance.copy(deep=True))

    assert sorted(result.data_vars) == ["radiance", "wf"]
    np.testing.assert_allclose(result["radiance"], expected["radiance"])
    dims = ("x", "wavelength", "los", "stokes")
    assert result["wf"].dims == dims
    np.testing.assert_allclose(result["wf"], expected["wf"].transpose(*dims))


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
