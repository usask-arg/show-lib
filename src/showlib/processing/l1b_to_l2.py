from __future__ import annotations

import logging
from pathlib import Path

import sasktran2 as sk
import xarray as xr
from skretrieval.util import configure_log

from showlib.l1b.data import L1bDataSet
from showlib.l2.data import L2FileWriter, L2Profile
from showlib.l2.processing import SHOWFPRetrieval


def process_l1b_to_l2(l1b_file: Path, output_folder: Path, ils_path: Path):
    ils = xr.open_dataset(ils_path)

    l1b_data = L1bDataSet(l1b_file)

    por_data = xr.open_dataset(l1b_data.l2_por_path)

    out_file = output_folder.joinpath(l1b_data.l2_path.stem + ".nc")

    logging.info("Processing %s to %s", l1b_file.stem, out_file.stem)

    low_alt = 11000

    if not out_file.exists():
        l2s = []
        for image in range(len(l1b_data.ds.time)):
            l1b_image = l1b_data.image(image, low_alt=low_alt, high_alt=20000)
            por_image = por_data.isel(time=image)

            ret = SHOWFPRetrieval(
                l1b_image,
                ils,
                por_data=por_image,
                low_alt=low_alt,
                minimizer="rodgers",
                target_kwargs={
                    "rescale_state_space": False,
                },
                rodgers_kwargs={
                    "lm_damping_method": "fletcher",
                    "lm_damping": 0.1,
                    "max_iter": 10,
                    "lm_change_factor": 10,
                    "iterative_update_lm": True,
                    "retreat_lm": True,
                    "apply_cholesky_scaling": False,
                    "convergence_factor": 1,
                    "convergence_check_method": "dcost",
                },
                scipy_kwargs={
                    "loss": "linear",
                    "include_bounds": False,
                    "max_nfev": 100,
                    "x_scale": "jac",
                    "tr_solver": "lsmr",
                    "method": "lm",
                    "xtol": 1e-6,
                },
                state_kwargs={
                    "absorbers": {
                        "h2o": {
                            "prior_influence": 1e0,
                            "tikh_factor": 2.5e-2,
                            "log_space": True,
                            "min_value": 0,
                            "max_value": 1e-1,
                            "prior": {"type": "mipas", "value": 1e-6},
                        },
                    },
                    "albedo": {
                        "min_wavelength": 1300,
                        "max_wavelength": 1500,
                        "wavelength_res": 100,
                        "initial_value": 0.01,
                        "prior_influence": 1e-6,
                        "tikh_factor": 1e6,
                    },
                    "splines": {
                        "los": {
                            "min_wavelength": 1363,
                            "max_wavelength": 1370,
                            "num_knots": 5,
                            "smoothing": 0,
                            "order": 2,
                            "type": "global",
                            "enabled": True,
                        }
                    },
                    "aerosols": {},
                    "scale_factors": {
                        "los": {
                            "type": "poly",
                            "order": 2,
                        },
                    },
                    "shifts": {
                        "wavelength": {
                            "type": "wavelength",
                            "enabled": True,
                        }
                    },
                },
                engine_kwargs={
                    "num_threads": 1,
                    "num_streams": 2,
                    "multiple_scatter_source": sk.MultipleScatterSource.DiscreteOrdinates,
                },
            )

            results = ret.retrieve()

            lat_15 = float(
                results["retrieved"]["tangent_latitude"].interp(tangent_altitude=15000)
            )
            lon_15 = float(
                results["retrieved"]["tangent_longitude"].interp(tangent_altitude=15000)
            )

            l2s.append(
                L2Profile(
                    altitude_m=results["retrieved"].altitude.to_numpy(),
                    h2o_vmr=results["retrieved"].h2o_vmr.to_numpy(),
                    h2o_vmr_1sigma=results["retrieved"].h2o_vmr_1sigma.to_numpy(),
                    h2o_vmr_prior=results["retrieved"].h2o_por.to_numpy(),
                    tropopause_altitude=float(results["retrieved"].tropopause_altitude),
                    tangent_latitude=results["retrieved"].tangent_latitude.to_numpy(),
                    tangent_longitude=results["retrieved"].tangent_longitude.to_numpy(),
                    averaging_kernel=results["retrieved"].averaging_kernel.to_numpy(),
                    latitude=lat_15,
                    longitude=lon_15,
                    time=l1b_image.ds.time,
                )
            )
        writer = L2FileWriter(l2s)

        writer.save(out_file)


if __name__ == "__main__":
    configure_log()
    in_folder = Path(
        "/datastore/root/research_projects/SHOW/er2_2023/data/science_flight_1_testing/l1b/"
    )

    for file in in_folder.glob("HAWC*"):
        if file.suffix == ".nc":
            process_l1b_to_l2(
                file,
                Path(
                    "/datastore/root/research_projects/SHOW/er2_2023/data/science_flight_1_testing/l2/",
                ),
                Path(
                    "/datastore/root/research_projects/SHOW/er2_2023/data/science_flight_1_testing/calibration/ils.nc"
                ),
            )
