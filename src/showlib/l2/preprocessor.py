from __future__ import annotations

import logging

import numpy as np
import sasktran2 as sk
import xarray as xr
from skretrieval.core.lineshape import UserLineShape
from skretrieval.retrieval.measvec import MeasurementVector, select
from skretrieval.retrieval.processing import Retrieval

from showlib.l1b.data import L1bDataSet
from showlib.l2.ancillary import Ancillary
from showlib.l2.showmodel import SHOWForwardModel


def l2_preprocessor(
    l1b_data: L1bDataSet,
    por_data: xr.Dataset,
    cal_db: xr.Dataset,
    process_slice=None,
    **kwargs,
):
    if process_slice is None:
        process_slice = range(len(l1b_data.ds.time))

    wvnum = kwargs.get("preprocessor_wvnum", 7317.8)

    upper_bound = kwargs.get("upper_bound", 30000)

    l2s = []
    for image in process_slice:
        msg = f"Starting preprocessor for image {image}"
        logging.info(msg)
        l1b_image = l1b_data.image(
            image,
            low_alt=0,
            high_alt=upper_bound,
            row_reduction=kwargs.get("row_reduction", 1),
            low_wvnumber_filter=kwargs.get("low_wvnumber_filter", 0),
            high_wvnumber_filter=kwargs.get("high_wvnumber_filter", 1e10),
            override_los=kwargs.get("override_los", False),
        )
        por_image = por_data.isel(time=image)

        good = ~np.isnan(por_image.pressure.to_numpy())
        anc = Ancillary(
            por_image.altitude.to_numpy()[good],
            por_image.pressure.to_numpy()[good],
            por_image.temperature.to_numpy()[good],
        )

        minimizer_kwargs = {
            "method": "trf",
            "xtol": 1e-15,
            "include_bounds": True,
            "max_nfev": 20,
            "ftol": 1e-6,
        }

        def ls_fn(w):
            if "sample_wavenumber" in cal_db:
                return UserLineShape(
                    cal_db.hires_wavenumber.to_numpy(),
                    cal_db.sel(sample_wavenumber=1e7 / w, method="nearest")[
                        "ils"
                    ].to_numpy(),
                    False,
                    integration_fraction=0.97,
                )
            return UserLineShape(
                cal_db.hires_wavenumber.to_numpy(),
                cal_db["ils"].to_numpy(),
                True,
                integration_fraction=0.97,
            )

        ret = Retrieval(
            observation=l1b_image,
            forward_model_cfg={
                "meas": {
                    "kwargs": {
                        "lineshape_fn": ls_fn,
                        "spectral_native_coordinate": "wavenumber_cminv",
                        "model_res_cminv": 0.01,
                    },
                    "class": SHOWForwardModel,
                },
            },
            minimizer="scipy",
            ancillary=anc,
            l1_kwargs={},
            model_kwargs={
                "num_threads": kwargs.get("num_threads", 1),
                "los_refraction": kwargs.get("los_refraction", True),
                "multiple_scatter_source": sk.MultipleScatterSource.DiscreteOrdinates,
                "num_streams": 2,
            },
            minimizer_kwargs=minimizer_kwargs,
            target_kwargs={},
            measvec={
                "meas": MeasurementVector(
                    lambda l1, ctxt, **kwargs: select(  # noqa: ARG005
                        l1, wavenumber=wvnum, method="nearest", **kwargs
                    ),
                    sample_fn=lambda samples: {
                        k: v.sel(wavenumber=wvnum, method="nearest")
                        for k, v in samples.items()
                    },
                )
            },
            state_kwargs={
                "absorbers": {
                    "h2o": {
                        "prior_influence": 1e6,
                        "tikh_factor": 2.5e3,
                        "log_space": False,
                        "min_value": 0,
                        "max_value": 1e-1,
                        "prior": {"type": "mipas", "value": 1e-6},
                    },
                },
                "aerosols": {
                    "stratospheric_aerosol": {
                        "type": "extinction_profile",
                        "nominal_wavelength": 1350,
                        "scale_factor": 1,
                        "retrieved_quantities": {
                            "extinction_per_m": {
                                "prior_influence": 1e-1,
                                "tikh_factor": 1e1,
                                "min_value": 0,
                                "max_value": 1e-3,
                            },
                        },
                        "prior": {
                            "extinction_per_m": {"type": "testing"},
                        },
                    },
                },
                "altitude_grid": kwargs["altitude_grid"],
            },
        )

        results = ret.retrieve()

        l1 = results["meas_l1"]["meas"]

        meas_ratio = (
            l1.data.sel(wavenumber=wvnum, method="nearest")
            / results["simulated_l1"]["meas"].data
        )["radiance"]

        # Smooth the ratio
        meas_ratio = meas_ratio.rolling(los=5, center=True).mean()

        # Find altitudes where abs(1 - ratio) > 0.1
        bad = np.abs(1 - meas_ratio) > 0.1
        bad_alt = l1.data.tangent_altitude.isel(los=bad.to_numpy().flatten())

        try:
            estimated_low_alt = bad_alt.max()
        except Exception:
            estimated_low_alt = 8000.0

        aero_prof = results["state"][
            "stratospheric_aerosol_extinction_per_m"
        ].to_numpy()
        # Set all values above the max tangent altitude to be 0
        aero_prof[
            results["state"].altitude.to_numpy() > float(l1.data.tangent_altitude.max())
        ] = 0

        l2s.append({"aerosol": aero_prof, "low_alt": float(estimated_low_alt)})

    return l2s
