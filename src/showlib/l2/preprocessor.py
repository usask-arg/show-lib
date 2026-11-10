from __future__ import annotations

import copy
import logging

import numpy as np
import xarray as xr
from skretrieval.retrieval.measvec import MeasurementVector, select
from skretrieval.retrieval.processing import Retrieval

from showlib.l1b.data import L1bDataSet
from showlib.l2.ancillary import Ancillary
from showlib.l2.config import altitude_grid, engine_kwargs
from showlib.l2.lineshape import SHOWLineShapes
from showlib.l2.showmodel import SHOWForwardModel


def l2_preprocessor(
    l1b_data: L1bDataSet,
    por_data: xr.Dataset,
    line_shapes: SHOWLineShapes,
    config: dict,
    process_slice=None,
):
    if process_slice is None:
        process_slice = range(len(l1b_data.ds.time))

    wvnum = config["preprocessor"]["wavenumber_cminv"]
    low_altitude = config["preprocessor"]["low_altitude"]
    observation = config["observation"]
    upper_bound = observation["upper_bound_m"]

    l2s = []
    for image in process_slice:
        msg = f"Starting preprocessor for image {image}"
        logging.info(msg)
        l1b_image = l1b_data.image(
            image,
            low_alt=0,
            high_alt=upper_bound,
            row_reduction=observation["row_reduction"],
            low_wvnumber_filter=observation["low_wavenumber_filter_cminv"],
            high_wvnumber_filter=observation["high_wavenumber_filter_cminv"],
            override_los=observation["override_los"],
        )
        por_image = por_data.isel(time=image)

        good = ~np.isnan(por_image.pressure.to_numpy())
        anc = Ancillary(
            por_image.altitude.to_numpy()[good],
            por_image.pressure.to_numpy()[good],
            por_image.temperature.to_numpy()[good],
        )

        state_kwargs = copy.deepcopy(config["preprocessor"]["state"])
        state_kwargs["altitude_grid"] = altitude_grid(config)

        ret = Retrieval(
            observation=l1b_image,
            forward_model_cfg={
                "meas": {
                    "kwargs": {
                        "line_shapes": line_shapes,
                        "rows": l1b_image.retrieval_rows(),
                    },
                    "class": SHOWForwardModel,
                },
            },
            minimizer="scipy",
            ancillary=anc,
            l1_kwargs={},
            model_kwargs=engine_kwargs(config, "preprocessor"),
            minimizer_kwargs=dict(config["minimizer"]),
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
            state_kwargs=state_kwargs,
        )

        results = ret.retrieve()

        l1 = results["meas_l1"]["meas"]

        meas_ratio = (
            l1.data.sel(wavenumber=wvnum, method="nearest")
            / results["simulated_l1"]["meas"].data
        )["radiance"]

        # Smooth the ratio
        meas_ratio = meas_ratio.rolling(
            los=low_altitude["smoothing_los"], center=True
        ).mean()

        # Find altitudes where abs(1 - ratio) > 0.1
        bad = np.abs(1 - meas_ratio) > low_altitude["ratio_threshold"]
        bad_alt = l1.data.tangent_altitude.isel(los=bad.to_numpy().flatten())

        try:
            estimated_low_alt = bad_alt.max()
        except Exception:
            estimated_low_alt = low_altitude["fallback_m"]

        if "aerosol_extinction" in por_data:
            aero_prof = por_data["aerosol_extinction"].isel(time=image).interp(altitude=results["state"].altitude.to_numpy()).to_numpy()
        else:
            aero_prof = results["state"][
                "stratospheric_aerosol_extinction_per_m"
            ].to_numpy()
        # Set all values above the max tangent altitude to be 0
        #aero_prof[
        #    results["state"].altitude.to_numpy() > float(l1.data.tangent_altitude.max())
        #] = 0

        l2s.append({"aerosol": aero_prof, "low_alt": float(estimated_low_alt)})

    return l2s
