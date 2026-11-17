from __future__ import annotations

import copy
import logging

import numpy as np
import sasktran2 as sk
import xarray as xr
from skretrieval.retrieval.measvec import MeasurementVector, select

from showlib.cal_db import CalibrationDatabase
from showlib.l1b.data import L1bDataSet
from showlib.l2.ancillary import Ancillary
from showlib.l2.config import altitude_grid, engine_kwargs
from showlib.l2.lineshape import SHOWLineShapes
from showlib.l2.optical import h2o_optical_property
from showlib.l2.retrieval import SHOWRetrieval
from showlib.l2.showmodel import SHOWForwardModel


def weakest_h2o_sample(
    cal_db: CalibrationDatabase,
    row: int = 0,
    min_transmission: float = 0.5,
    pressure_pa: float = 12000.0,
    temperature_k: float = 216.65,
) -> float:
    """The in-band sample wavenumber [cm-1] with the least H2O absorption.

    Uses the retrieval's HITRAN H2O cross sections at a lower-stratosphere
    reference state (by default about 15 km in the US standard atmosphere), among
    the row's samples where the spectral transmission is at least
    ``min_transmission``. Suitable for preprocessor.wavenumber_cminv.
    """
    samples = cal_db.sample_wavenumber(row)
    samples = samples[cal_db.spectral_transmission(samples) >= min_transmission]
    if len(samples) == 0:
        msg = f"No samples in row {row} have transmission >= {min_transmission}"
        raise ValueError(msg)

    altitudes = np.array([0.0, 1.0, 2.0])
    geometry = sk.Geometry1D(
        0.5,
        0.0,
        6372e3,
        altitudes,
        sk.InterpolationMethod.LinearInterpolation,
        sk.GeometryType.Spherical,
    )
    atmo = sk.Atmosphere(geometry, sk.Config(), wavelengths_nm=1e7 / samples)
    atmo.pressure_pa = np.full(len(altitudes), pressure_pa)
    atmo.temperature_k = np.full(len(altitudes), temperature_k)
    cross_section = np.asarray(
        h2o_optical_property().atmosphere_quantities(atmo).extinction
    )
    return float(samples[np.argmin(cross_section[0])])


def lowest_usable_altitude(
    tangent_altitude: np.ndarray,
    ratio: np.ndarray,
    ratio_noise: np.ndarray,
    *,
    smoothing_los: int,
    ratio_threshold: float,
    noise_sigma: float,
    min_good_rows: int,
    fallback_m: float,
) -> float:
    """The altitude [m] above which the pre-fit's rows are usable.

    ``ratio`` is measured/modelled radiance per line of sight and ``ratio_noise``
    its 1-sigma noise. Both are averaged over a centred window of
    ``smoothing_los`` rows in altitude order (fewer at the ends), with the rows'
    noise taken as independent. A row is bad when its smoothed ratio differs from
    1 by more than ``ratio_threshold`` plus ``noise_sigma`` times its smoothed
    noise, or is missing (missing rows are left out of their neighbours' means).

    Searching up from the lowest row, the result is the altitude of the highest
    bad row below the first ``min_good_rows`` consecutive good rows, so rows
    strictly above it are used. ``fallback_m`` is returned when no bad row is
    below that run, or when there is no such run.
    """
    order = np.argsort(tangent_altitude)
    altitude = np.asarray(tangent_altitude, dtype=float)[order]
    ratio = np.asarray(ratio, dtype=float)[order]
    noise = np.asarray(ratio_noise, dtype=float)[order]

    present = np.isfinite(ratio) & np.isfinite(noise)
    start = smoothing_los - 1 - smoothing_los // 2

    def centred_sum(values):
        values = np.where(present, values, 0.0)
        summed = np.convolve(values, np.ones(smoothing_los), mode="full")
        return summed[start : start + len(values)]

    with np.errstate(invalid="ignore", divide="ignore"):
        count = centred_sum(np.ones_like(ratio))
        smoothed = centred_sum(ratio) / count
        smoothed_noise = np.sqrt(centred_sum(noise**2)) / count

    good = present & (
        np.abs(1 - smoothed) <= ratio_threshold + noise_sigma * smoothed_noise
    )

    for first in range(len(good) - min_good_rows + 1):
        if good[first : first + min_good_rows].all():
            return fallback_m if first == 0 else float(altitude[first - 1])

    logging.warning(
        "No run of %d usable rows in the pre-fit, using the fallback %.0f m",
        min_good_rows,
        fallback_m,
    )
    return fallback_m


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
    if wvnum == "auto":
        wvnum = weakest_h2o_sample(line_shapes.cal_db)
        logging.info("Preprocessor wavenumber: %.3f cm-1", wvnum)
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
            row_averaging=observation["row_averaging"],
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

        ret = SHOWRetrieval(
            observation=l1b_image,
            forward_model_cfg={
                "meas": {
                    "kwargs": {
                        "line_shapes": line_shapes,
                        "rows": l1b_image.retrieval_rows(),
                        # Model only the selected sample's line-shape window
                        "line_shape_tolerance": config["preprocessor"][
                            "line_shape_tolerance"
                        ],
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

        measured = l1.data.sel(wavenumber=wvnum, method="nearest")
        modelled = results["simulated_l1"]["meas"].data["radiance"]
        estimated_low_alt = lowest_usable_altitude(
            l1.data.tangent_altitude.to_numpy(),
            (measured["radiance"] / modelled).to_numpy().ravel(),
            (measured["radiance_noise"] / modelled).to_numpy().ravel(),
            **low_altitude,
        )
        logging.info("Lowest usable altitude: %.0f m", estimated_low_alt)

        if "aerosol_extinction" in por_data:
            aero_prof = (
                por_data["aerosol_extinction"]
                .isel(time=image)
                .interp(altitude=results["state"].altitude.to_numpy())
                .to_numpy()
            )
        else:
            aero_prof = results["state"][
                "stratospheric_aerosol_extinction_per_m"
            ].to_numpy()
        # Set all values above the max tangent altitude to be 0
        # aero_prof[
        #    results["state"].altitude.to_numpy() > float(l1.data.tangent_altitude.max())
        # ] = 0

        l2s.append({"aerosol": aero_prof, "low_alt": estimated_low_alt})

    return l2s
