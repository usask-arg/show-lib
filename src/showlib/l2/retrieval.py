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
from skretrieval.retrieval.processing import Retrieval
from skretrieval.retrieval.statevector.constituent import StateVectorElementConstituent

from showlib.l2.optical import h2o_optical_property


class SHOWRetrieval(Retrieval):
    """skretrieval Retrieval with the SHOW optical properties and state types."""


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
    else:
        aero_const = sk.test_util.scenarios.test_aerosol_constituent(native_alt_grid)

        ext = copy.copy(aero_const.extinction_per_m)

    # Floor at a fraction of the peak, so levels where the prior is zero can still
    # be retrieved and the profile stays positive for log space
    floor = cfg.get("floor_fraction_of_peak", 0.0) * np.max(ext)
    ext = np.maximum(ext, max(floor, 1e-15))

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
