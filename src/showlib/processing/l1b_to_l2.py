from __future__ import annotations

import logging
from pathlib import Path

import xarray as xr
from skretrieval.util import configure_log

from showlib.config import ils_folder
from showlib.l1b.data import L1bDataSet
from showlib.l2.data import L2FileWriter, L2Profile
from showlib.l2.processing import SHOWFPRetrieval


def process_l1b_to_l2(l1b_file: Path, output_folder: Path):
    ils = xr.open_dataset(
        ils_folder().joinpath("ils_characterization_hann_2024_02_08.nc")
    )

    l1b_data = L1bDataSet(l1b_file)
    por_data = xr.open_dataset(l1b_data.l2_por_path)

    out_file = output_folder.joinpath(l1b_data.l2_path.stem + +".nc")

    logging.info("Processing %s to %s", l1b_file.stem, out_file.stem)

    if not out_file.exists():
        l2s = []
        for image in range(len(l1b_data.ds.time)):
            l1b_image = l1b_data.image(image)
            por_image = por_data.isel(time=image)

            ret = SHOWFPRetrieval(
                l1b_image,
                ils,
                por_data=por_image,
                low_alt=5000,
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
                        }
                    },
                    "aerosols": {},
                    "scale_factors": {
                        "los": {
                            "type": "poly",
                            "order": 2,
                        }
                    },
                    "shifts": {"wavelength": {"type": "wavelength"}},
                },
                engine_kwargs={"num_threads": 8},
            )

            results = ret.retrieve()
            l2s.append(
                L2Profile(
                    altitude_m=results["retrieved"].altitude.to_numpy(),
                    h2o_vmr=results["retrieved"].h2o_vmr.to_numpy(),
                    latitude=l1b_image.lat,
                    longitude=l1b_image.lon,
                    time=l1b_image.ds.time,
                )
            )
        writer = L2FileWriter(l2s)

        writer.save(out_file)


if __name__ == "__main__":
    configure_log()
    in_folder = Path(
        "/Users/dannyz/OneDrive - University of Saskatchewan/SHOW/er2_2023/sci_flight/l1b_mag/"
    )

    for file in in_folder.iterdir():
        if file.suffix == ".nc":
            process_l1b_to_l2(
                file,
                Path(
                    "/Users/dannyz/OneDrive - University of Saskatchewan/SHOW/er2_2023/sci_flight/l2_mag"
                ),
            )
