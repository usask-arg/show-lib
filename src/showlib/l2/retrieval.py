"""The SHOW L2 retrieval class, with SHOW's optical properties and state types.

They are registered on SHOWRetrieval rather than skretrieval's shared Retrieval,
which other packages (e.g. aliprocessing) register the same names on in the same
process. The isolation needs an skretrieval release with per-subclass registries;
with an older one the registrations still work but are shared.
"""

from __future__ import annotations

import copy

import numpy as np
import sasktran2 as sk
import skretrieval.retrieval.prior as prior
import xarray as xr
from skretrieval.retrieval.processing import Retrieval
from skretrieval.retrieval.statevector.altitude import AltitudeNativeStateVector
from skretrieval.retrieval.statevector.constituent import StateVectorElementConstituent

from showlib.l2.optical import h2o_optical_property


class SHOWStateVector(AltitudeNativeStateVector):
    """AltitudeNativeStateVector that maps SASKTRAN2 radiances with less memory.

    The weighting functions on the model grid (state size by model wavenumbers by
    lines of sight) are the retrieval's largest arrays. skretrieval holds every
    element's weighting functions and their concatenation at once, and modifies
    SASKTRAN2's own weighting functions before dropping them. This writes each
    element's weighting functions into one array as they are computed, and drops
    SASKTRAN2's before the radiance is modified. The result is the same.
    """

    def post_process_sk2_radiances(self, radiance: xr.Dataset) -> xr.Dataset:
        elements = [element for element in self.state_elements if element.enabled]
        dims = radiance["radiance"].dims
        sizes = [len(element.state()) for element in elements]
        wf = np.empty((sum(sizes), *radiance["radiance"].shape))
        start = 0
        for element, size in zip(elements, sizes, strict=True):
            element_wf = element.propagate_wf(radiance).transpose("x", *dims)
            if element_wf.sizes["x"] != size:
                msg = (
                    f"{type(element).__name__} has {size} state values but "
                    f"{element_wf.sizes['x']} weighting functions"
                )
                raise ValueError(msg)
            wf[start : start + size] = element_wf.to_numpy()
            start += size
            del element_wf

        radiance = radiance.drop_vars([key for key in radiance if key.startswith("wf")])
        for element in elements:
            radiance = element.modify_input_radiance(radiance)

        radiance["wf"] = (["x", *dims], wf)
        return radiance


class SHOWRetrieval(Retrieval):
    """skretrieval Retrieval with the SHOW optical properties and state types."""

    def _construct_state_vector(self):
        state_vector = super()._construct_state_vector()
        return SHOWStateVector(state_vector.altitude_grid, **state_vector.sv)

    def retrieve(
        self,
        enabled_state_elements: list[str] | None = None,
        enabled_measurement_vectors: list[str] | None = None,
    ):
        """Retrieve, and also describe the state with the solution covariance.

        skretrieval's ``state`` gives each element's ``_1sigma_error`` from the
        measurement-noise covariance G S_e G^T. ``solution_state`` is the same
        description with the solution covariance (K^T S_e^-1 K + R)^-1 instead, so
        its ``_1sigma_error`` variables are the solution errors (see l2.yaml).
        """
        sv = self._state_vector.sv
        # The elements enabled during the retrieval; skretrieval re-enables them
        # all afterwards, which would misalign the covariance slices
        enabled = {
            name: (
                element.enabled
                if enabled_state_elements is None
                else name in enabled_state_elements
            )
            for name, element in sv.items()
        }
        results = super().retrieve(enabled_state_elements, enabled_measurement_vectors)

        after = {name: element.enabled for name, element in sv.items()}
        try:
            for name, element in sv.items():
                element.enabled = enabled[name]
            results["solution_state"] = self._state_vector.describe(
                {
                    "error_covariance_from_noise": results["minimizer"][
                        "solution_covariance"
                    ]
                }
            )
        finally:
            for name, element in sv.items():
                element.enabled = after[name]
        return results


@SHOWRetrieval.register_optical_property("h2o")
def h2o_ret_optical_property():
    return h2o_optical_property()


@SHOWRetrieval.register_optical_property("stratospheric_aerosol")
def stratospheric_aerosol_optical_property():
    refrac = sk.mie.refractive.H2SO4()
    dist = sk.mie.distribution.LogNormalDistribution().freeze(
        mode_width=1.6, median_radius=80
    )

    return sk.database.MieDatabase(dist, refrac, np.arange(1300, 1401.0, 5.0))


# In log space, a lower bound at or below zero becomes this finite floor: log(0) is
# -inf, and skretrieval's scipy minimizer multiplies the bounds by a matrix, which
# turns an infinite bound into NaN for every state element.
_LOG_SPACE_MIN_EXTINCTION_PER_M = 1e-30


@SHOWRetrieval.register_state("aerosols", "extinction_profile_2")
def aerosol_extinction_profile(self, name: str, native_alt_grid: np.array, cfg: dict):
    if cfg.get("prior_state") is not None:
        ext = np.array(cfg["prior_state"], dtype=float)
        if not np.any(np.isfinite(ext)):
            msg = "Aerosol prior_state has no finite values"
            raise ValueError(msg)
    else:
        aero_const = sk.test_util.scenarios.test_aerosol_constituent(native_alt_grid)

        ext = copy.copy(aero_const.extinction_per_m)

    # Floor at a fraction of the peak, so levels where the prior is zero can still
    # be retrieved and the profile stays positive for log space. Non-finite levels
    # (e.g. where a POR aerosol profile does not reach) take the floor too.
    floor = max(cfg.get("floor_fraction_of_peak", 0.0) * np.nanmax(ext), 1e-15)
    ext = np.where(np.isfinite(ext), np.maximum(ext, floor), floor)

    scale_factor = cfg.get("scale_factor", 1)
    log_space = cfg.get("log_space", False)

    secondary_kwargs = {
        name: np.ones_like(native_alt_grid) * cfg["prior"][name]["value"]
        for name in cfg["prior"]
        if name != "extinction_per_m"
    }

    db = self._optical_property(name)

    aero_const = sk.constituent.ExtinctionScatterer(
        db,
        native_alt_grid,
        ext * scale_factor,
        cfg.get("nominal_wavelength", 745),
        "extend",
        **secondary_kwargs,
    )

    sv_ele = StateVectorElementConstituent(
        aero_const,
        f"{name}",
        cfg["retrieved_quantities"].keys(),
        min_value={
            name: (
                _LOG_SPACE_MIN_EXTINCTION_PER_M
                if log_space and val["min_value"] <= 0
                else val["min_value"]
            )
            for name, val in cfg["retrieved_quantities"].items()
        },
        max_value={
            name: val["max_value"] for name, val in cfg["retrieved_quantities"].items()
        },
        prior={
            name: val["tikh_factor"]
            * prior.VerticalTikhonov(
                1,
                # Prior states are in state-vector units, so log in log space
                prior_state=(np.log if log_space else np.asarray)(
                    secondary_kwargs.get(name, ext * scale_factor)
                ),
            )
            + val["prior_influence"] * prior.ConstantDiagonalPrior()
            for name, val in cfg["retrieved_quantities"].items()
        },
        log_space=log_space,
    )

    sv_ele.enabled = cfg.get("enabled", True)

    if cfg.get("initial_guess") is not None:
        sv_ele.update_state(cfg["initial_guess"])

    return sv_ele
