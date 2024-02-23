from __future__ import annotations

from copy import copy

import numpy as np
import sasktran2 as sk2
import xarray as xr
from skretrieval.retrieval.rodgers import Rodgers
from skretrieval.retrieval.scipy import SciPyMinimizer, SciPyMinimizerGrad
from skretrieval.retrieval.statevector.constituent import StateVectorElementConstituent

from showlib.l1b.data import L1bImageBase

from .ancillary import SHOWAncillary
from .shifts import AltitudeShift, BandShifts
from .showmodel import SHOWFPForwardModel
from .spline import (
    AddFactors,
    MultiplicativeSpline,
    MultiplicativeSplineOne,
    ScaleFactorsPoly,
)
from .statevector import SHOWStateVector
from .target import SHOWFPTarget


class SHOWFPRetrieval:
    def __init__(
        self,
        l1b_image: L1bImageBase,
        ils: xr.Dataset,
        minimizer="rodgers",
        por_data: xr.Dataset | None = None,
        rodgers_kwargs: dict | None = None,
        scipy_kwargs: dict | None = None,
        target_kwargs: dict | None = None,
        state_kwargs: dict | None = None,
        engine_kwargs: dict | None = None,
        low_alt: float = 0,
        top_retrieval_alt: float = 22000,
        top_atmosphere_alt: float = 50000,
        fine_altitude_resolution: float = 250,
        coarse_altitude_resolution: float = 1000,
        **kwargs,
    ) -> None:
        if engine_kwargs is None:
            engine_kwargs = {}
        if state_kwargs is None:
            state_kwargs = {}
        if target_kwargs is None:
            target_kwargs = {}
        if scipy_kwargs is None:
            scipy_kwargs = {}
        if rodgers_kwargs is None:
            rodgers_kwargs = {}
        self._options = kwargs
        self._l1b = l1b_image
        self._skl1 = self._l1b.skretrieval_l1()
        self._ils = ils
        self._minimizer = minimizer
        self._rodgers_kwargs = rodgers_kwargs
        self._target_kwargs = target_kwargs
        self._scipy_kwargs = scipy_kwargs
        self._state_kwargs = state_kwargs
        self._engine_kwargs = engine_kwargs
        self._por_data = por_data
        self._native_alt_grid = np.unique(
            np.concatenate(
                (
                    np.arange(low_alt, top_retrieval_alt, fine_altitude_resolution),
                    np.arange(
                        top_retrieval_alt,
                        top_atmosphere_alt,
                        coarse_altitude_resolution,
                    ),
                )
            )
        )

        self._anc = self._construct_ancillary()
        self._state_vector = self._construct_state_vector()
        self._forward_model = self._construct_forward_model()

    def _construct_forward_model(self):
        engine_config = sk2.Config()
        for key, val in self._engine_kwargs.items():
            setattr(engine_config, key, val)

        return SHOWFPForwardModel(
            self._l1b,
            self._ils,
            self._native_alt_grid,
            self._state_vector,
            self._anc,
            engine_config,
        )

    def _const_from_mipas(
        self,
        alt_grid,
        species_name,
        optical,
        prior_infl=1e-2,
        tikh=1e8,
        log_space=False,
        min_val=0,
        max_val=1,
    ):
        const = sk2.climatology.mipas.constituent(species_name, optical)

        new_vmr = np.interp(alt_grid, const._altitudes_m, const.vmr)

        new_const = sk2.constituent.VMRAltitudeAbsorber(
            optical, alt_grid, new_vmr, out_of_bounds_mode="extend"
        )

        if species_name == "so2":
            new_const.vmr[:] = 1e-10

        min_val = 1e-40 if log_space else min_val

        return StateVectorElementConstituent(
            new_const,
            species_name,
            ["vmr"],
            min_value={"vmr": min_val},
            max_value={"vmr": max_val},
            prior_influence={"vmr": prior_infl},
            first_order_tikh={"vmr": tikh},
            log_space=log_space,
        )

    def _optical_property(self, species_name: str):
        if species_name.lower() == "h2o":
            return sk2.database.HITRANDatabase(
                molecule="H2O",
                start_wavenumber=7295,
                end_wavenumber=7340,
                wavenumber_resolution=0.01,
                reduction_factor=1,
                backend="sasktran_legacy",
            )
        return None

    def _construct_state_vector(self):
        absorbers = {}

        for name, options in self._state_kwargs["absorbers"].items():
            if options["prior"]["type"] == "mipas":
                absorbers[name] = self._const_from_mipas(
                    self._native_alt_grid,
                    name,
                    self._optical_property(name),
                    tikh=options["tikh_factor"],
                    prior_infl=options["prior_influence"],
                    log_space=options["log_space"],
                    min_val=options["min_value"],
                    max_val=options["max_value"],
                )
            elif options["prior"]["type"] == "constant":
                const = sk2.constituent.VMRAltitudeAbsorber(
                    self._optical_property(name),
                    self._native_alt_grid,
                    np.ones_like(self._native_alt_grid)
                    * float(options["prior"]["value"]),
                )
                absorbers[name] = StateVectorElementConstituent(
                    const,
                    name,
                    ["vmr"],
                    min_value={"vmr": options["min_value"]},
                    max_value={"vmr": options["max_value"]},
                    prior_influence={"vmr": options["prior_influence"]},
                    first_order_tikh={"vmr": options["tikh_factor"]},
                    log_space=options["log_space"],
                )

        surface = {}
        if "albedo" in self._state_kwargs:
            options = self._state_kwargs["albedo"]

            albedo_wavel = np.arange(
                options["min_wavelength"],
                options["max_wavelength"],
                options["wavelength_res"],
            )
            albedo_start = np.ones_like(albedo_wavel) * options["initial_value"]

            albedo_const = sk2.constituent.LambertianSurface(albedo_start, albedo_wavel)
            surface["albedo"] = StateVectorElementConstituent(
                albedo_const,
                "albedo",
                ["albedo"],
                min_value={"albedo": 0},
                max_value={"albedo": 1},
                prior_influence={"albedo": options["prior_influence"]},
                first_order_tikh={"albedo": options["tikh_factor"]},
                log_space=False,
            )

        aerosols = {}
        for name, aerosol in self._state_kwargs["aerosols"].items():
            if aerosol["type"] == "extinction_profile":
                aero_const = sk2.test_util.scenarios.test_aerosol_constituent(
                    self._native_alt_grid
                )

                ext = copy(aero_const.extinction_per_m)
                ext[ext == 0] = 1e-15

                secondary_kwargs = {
                    name: np.ones_like(self._native_alt_grid)
                    * aerosol["prior"][name]["value"]
                    for name in aerosol["prior"]
                    if name != "extinction_per_m"
                }

                aero_const = sk2.constituent.ExtinctionScatterer(
                    sk2.optical.database.OpticalDatabaseGenericScatterer(
                        sk2.appconfig.database_root().joinpath(
                            "cross_sections/mie/sulfate_test.nc"
                        )
                    ),
                    self._native_alt_grid,
                    ext,
                    745,
                    "extend",
                    **secondary_kwargs,
                )

                aerosols[f"aerosol_{name}"] = StateVectorElementConstituent(
                    aero_const,
                    f"aerosol_{name}",
                    aerosol["retrieved_quantities"].keys(),
                    min_value={
                        name: val["min_value"]
                        for name, val in aerosol["retrieved_quantities"].items()
                    },
                    max_value={
                        name: val["max_value"]
                        for name, val in aerosol["retrieved_quantities"].items()
                    },
                    prior_influence={
                        name: val["prior_influence"]
                        for name, val in aerosol["retrieved_quantities"].items()
                    },
                    first_order_tikh={
                        name: val["tikh_factor"]
                        for name, val in aerosol["retrieved_quantities"].items()
                    },
                    log_space=False,
                )

        splines = {}
        for name, spline in self._state_kwargs["splines"].items():
            if spline["type"] == "los":
                splines[f"spline_{name}"] = MultiplicativeSpline(
                    len(self._l1b.ds.los),
                    spline["min_wavelength"],
                    spline["max_wavelength"],
                    spline["num_knots"],
                    s=spline["smoothing"],
                    order=spline["order"],
                )
                splines[f"spline_{name}"].enabled = spline.get("enabled", True)
            else:
                splines[f"spline_{name}"] = MultiplicativeSplineOne(
                    spline["min_wavelength"],
                    spline["max_wavelength"],
                    spline["num_knots"],
                    s=spline["smoothing"],
                    order=spline["order"],
                )
                splines[f"spline_{name}"].enabled = spline.get("enabled", True)

        scales = {}
        for name, scale in self._state_kwargs["scale_factors"].items():
            if scale["type"] == "poly":
                scales[f"scale_{name}"] = ScaleFactorsPoly(
                    len(self._l1b.ds.los), order=scale["order"]
                )
                scales[f"scale_{name}"].enabled = scale.get("enabled", True)
            if scale["type"] == "add":
                scales[f"add_{name}"] = AddFactors(
                    len(self._l1b.ds.los), order=scale["order"]
                )

        shifts = {}
        for name, shift in self._state_kwargs["shifts"].items():
            if shift["type"] == "wavelength":
                shifts[f"shift_{name}"] = BandShifts(len(self._l1b.ds.los))
                shifts[f"shift_{name}"].enabled = shift.get("enabled", True)
            if shift["type"] == "altitude":
                shifts[f"shift_{name}"] = AltitudeShift()

        return SHOWStateVector(
            **absorbers, **surface, **splines, **aerosols, **shifts, **scales
        )

    def _construct_ancillary(self):
        if self._por_data is not None:
            return SHOWAncillary(
                self._por_data["altitude"].to_numpy(),
                self._por_data["pressure"].to_numpy(),
                self._por_data["temperature"].to_numpy(),
            )
        lat = self._l1b.lat
        lon = self._l1b.lon

        import sasktran as sk

        msis = sk.MSIS90()

        pres = msis.get_parameter(
            "SKCLIMATOLOGY_PRESSURE_PA", lat, lon, self._native_alt_grid, 60275
        )
        temp = msis.get_parameter(
            "SKCLIMATOLOGY_TEMPERATURE_K", lat, lon, self._native_alt_grid, 60275
        )

        return SHOWAncillary(self._native_alt_grid, pres, temp)

    def _construct_target(self):
        return SHOWFPTarget(self._state_vector, **self._target_kwargs)

    def _construct_output(self, rodgers_output: dict):
        return rodgers_output

    def validate_wf(self, state_idx):
        dx = self._target.state_vector()[state_idx] * 1e-3

        old_x = copy(self._target.state_vector())
        new_x = copy(old_x)

        base_rad = self._forward_model.calculate_radiance()

        new_x[state_idx] += dx
        self._target.update_state(new_x)

        pert_rad = self._forward_model.calculate_radiance()

        self._target.update_state(old_x)

        num_wf = (  # noqa: F841
            pert_rad.data["radiance"] - base_rad.data["radiance"]
        ) / dx
        ana_wf = base_rad.data["wf"].isel(x=state_idx)  # noqa: F841

    def retrieve(self):
        if self._minimizer == "rodgers":
            minimizer = Rodgers(**self._rodgers_kwargs)
        elif self._minimizer == "scipy":
            minimizer = SciPyMinimizer(**self._scipy_kwargs)
        elif self._minimizer == "scipy_grad":
            minimizer = SciPyMinimizerGrad()

        results = minimizer.retrieve(
            self._l1b.skretrieval_l1(), self._forward_model, self._construct_target()
        )

        # Post process

        final_l1 = self._forward_model.calculate_radiance()
        meas_l1 = self._l1b.skretrieval_l1()

        results = {}

        results["minimizer"] = results
        results["meas_l1"] = meas_l1
        results["simulated_l1"] = final_l1

        results["retrieved"] = xr.Dataset()
        results["retrieved"].coords[
            "altitude"
        ] = self._forward_model._atmosphere.model_geometry.altitudes()

        for name in self._state_kwargs["absorbers"]:
            results["retrieved"][f"{name}_vmr"] = (
                ("altitude"),
                self._forward_model._atmosphere[name]._constituent.vmr,
            )
            results["retrieved"][f"{name}_vmr_prior"] = (
                ("altitude"),
                self._forward_model._atmosphere[name]._prior,
            )

        results["retrieved"].coords[
            "albedo_wavelength"
        ] = self._forward_model._atmosphere.wavelengths_nm
        results["retrieved"]["albedo"] = (
            ("albedo_wavelength"),
            self._forward_model._atmosphere.surface.albedo,
        )

        for name, aerosol in self._state_kwargs["aerosols"].items():
            for quantity in aerosol["retrieved_quantities"]:
                val = getattr(
                    self._forward_model._atmosphere[f"aerosol_{name}"]._constituent,
                    quantity,
                )
                results["retrieved"][f"aerosol_{name}_{quantity}"] = (("altitude"), val)

        return self._construct_output(results)
