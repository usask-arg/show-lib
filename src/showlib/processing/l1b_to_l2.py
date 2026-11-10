from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
import xarray as xr
from skretrieval.util import configure_log

from showlib.cal_db import CalibrationDatabase
from showlib.l1b.data import L1bDataSet
from showlib.l2.ancillary import Ancillary
from showlib.l2.config import altitude_grid, engine_kwargs, load_retrieval_config
from showlib.l2.data import L2FileWriter, L2Profile
from showlib.l2.lineshape import SHOWLineShapes
from showlib.l2.noise import noise_correlation_applies, whitened_measurement_vector
from showlib.l2.preprocessor import l2_preprocessor
from showlib.l2.retrieval import (  # noqa: F401  (old names, still importable here)
    SHOWRetrieval,
    aerosol_extinction_profile,
    h2o_ret_optical_property,
    stratospheric_aerosol_optical_property,
)
from showlib.l2.showmodel import SHOWForwardModel


def process_l1b_to_l2_file(
    l1b_file: Path, output_folder: Path, cal_db_path: Path, config: dict
):
    """Retrieve every image of an L1B granule; ``config`` from load_retrieval_config."""
    # Load in the data
    cal_db = CalibrationDatabase.from_file(cal_db_path)
    l1b_data = L1bDataSet.from_file(l1b_file)
    line_shapes = SHOWLineShapes.from_l1b(
        cal_db, l1b_data, model_res_cminv=config["model_res_cminv"]
    )
    por_data = xr.open_dataset(l1b_data.l2_por_path)

    # And get the output file name
    out_file = output_folder.joinpath(l1b_data.l2_path.stem + ".nc")

    if not out_file.exists():
        logging.info("Processing %s to %s", l1b_file.stem, out_file.stem)

        writer = L2FileWriter(
            process_l1b_to_l2(l1b_data, por_data, line_shapes, config)
        )

        writer.save(out_file)


def process_l1b_to_l2(
    l1b_data: L1bDataSet,
    por_data: xr.Dataset,
    line_shapes: SHOWLineShapes,
    config: dict,
    process_slice=None,
):
    grid = altitude_grid(config)
    observation = config["observation"]
    if process_slice is None:
        process_slice = range(len(l1b_data.ds.time))

    preprocessor_data = l2_preprocessor(
        l1b_data, por_data, line_shapes, config, process_slice
    )

    use_noise_correlation = config["retrieval"]["noise_correlation"]
    if use_noise_correlation and not noise_correlation_applies(l1b_data.ds.attrs):
        logging.warning(
            "Treating L1B samples as independent: noise correlation needs "
            "radiance_noise_model 'detector', not %r",
            l1b_data.ds.attrs.get("radiance_noise_model"),
        )
        use_noise_correlation = False

    upper_bound = observation["upper_bound_m"]
    configured_lower_bound = observation["lower_bound_m"]
    if configured_lower_bound is not None:
        if not np.isfinite(configured_lower_bound):
            msg = "lower_bound must be finite"
            raise ValueError(msg)
        if configured_lower_bound >= upper_bound:
            msg = "lower_bound must be less than upper_bound"
            raise ValueError(msg)

    l2s = []
    for i, image in enumerate(process_slice):
        preprocessor = preprocessor_data[i]
        retrieval_lower_bound = (
            preprocessor["low_alt"]
            if configured_lower_bound is None
            else float(configured_lower_bound)
        )

        l1b_image = l1b_data.image(
            image,
            low_alt=retrieval_lower_bound,
            high_alt=upper_bound,
            row_reduction=observation["row_reduction"],
            low_wvnumber_filter=observation["low_wavenumber_filter_cminv"],
            high_wvnumber_filter=observation["high_wavenumber_filter_cminv"],
            override_los=observation["override_los"],
        )
        por_image = por_data.isel(time=image)

        skl1 = l1b_image.skretrieval_l1()
        min_wv = float(skl1["meas"].data["wavenumber"].min())
        max_wv = float(skl1["meas"].data["wavenumber"].max())

        good = ~np.isnan(por_image.pressure.to_numpy())
        anc = Ancillary(
            por_image.altitude.to_numpy()[good],
            por_image.pressure.to_numpy()[good],
            por_image.temperature.to_numpy()[good],
        )

        state_kwargs = copy.deepcopy(config["retrieval"]["state"])
        state_kwargs["aerosols"]["stratospheric_aerosol"]["prior_state"] = preprocessor[
            "aerosol"
        ]
        state_kwargs["surface"]["lambertian_albedo"]["wavelengths"] = np.array(
            [1e7 / max_wv, 1e7 / min_wv]
        )
        state_kwargs["shifts"]["wavenumber"]["num_los"] = len(skl1["meas"].data.los)
        state_kwargs["splines"]["constant_los"]["low_wavelength_nm"] = 1e7 / max_wv
        state_kwargs["splines"]["constant_los"]["high_wavelength_nm"] = 1e7 / min_wv
        state_kwargs["altitude_grid"] = grid

        measvec = None
        if use_noise_correlation:
            measvec = {
                "measurement": whitened_measurement_vector(
                    skl1["meas"].data, line_shapes, l1b_image.retrieval_rows()
                )
            }

        ret = SHOWRetrieval(
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
            model_kwargs=engine_kwargs(config, "retrieval"),
            minimizer_kwargs=dict(config["minimizer"]),
            target_kwargs={},
            measvec=measvec,
            state_kwargs=state_kwargs,
        )

        results = ret.retrieve()

        ref_lat = l1b_image.reference_latitude()["meas"]
        ref_lon = l1b_image.reference_longitude()["meas"]

        l2s.append(
            L2Profile(
                altitude_m=results["state"].altitude.to_numpy(),
                h2o_vmr=results["state"].h2o_vmr.to_numpy(),
                h2o_vmr_1sigma=results["state"].h2o_vmr_1sigma_error.to_numpy(),
                h2o_vmr_prior=results["state"].h2o_vmr_prior.to_numpy(),
                tropopause_altitude=float(por_image.tropopause_altitude.to_numpy()),
                averaging_kernel=results["state"].h2o_vmr_averaging_kernel.to_numpy(),
                latitude=ref_lat,
                longitude=ref_lon,
                time=l1b_image.ds.time,
            )
        )
    return l2s


if __name__ == "__main__":
    configure_log()
    in_folder = Path(
        r"/Users/dannyz/OneDrive - University of Saskatchewan/SHOW/er2_2023/sci_flight/l1b_mag"
    )

    for file in in_folder.glob("HAWC*"):
        if file.suffix == ".nc":
            process_l1b_to_l2_file(
                file,
                Path(
                    r"/Users/dannyz/OneDrive - University of Saskatchewan/SHOW/er2_2023/sci_flight/l2",
                ),
                Path(r"/Users/dannyz/Documents/sds/ils.nc"),
                load_retrieval_config({"preprocessor": {"wavenumber_cminv": 7317.8}}),
            )
