from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import sasktran2 as sk
import xarray as xr
from skretrieval.core.lineshape import UserLineShape
from skretrieval.retrieval.measvec import MeasurementVector, select
from skretrieval.retrieval.processing import Retrieval
from skretrieval.util import configure_log

from showlib.l1b.data import L1bDataSet
from showlib.l2.ancillary import Ancillary
from showlib.l2.data import L2FileWriter, L2Profile
from showlib.l2.optical import h2o_optical_property
from showlib.l2.showmodel import SHOWForwardModel


@Retrieval.register_optical_property("h2o")
def h2o_ret_optical_property():
    return h2o_optical_property()


@Retrieval.register_optical_property("stratospheric_aerosol")
def stratospheric_aerosol_optical_property():
    refrac = sk.mie.refractive.H2SO4()
    dist = sk.mie.distribution.LogNormalDistribution().freeze(
        mode_width=1.6, median_radius=80
    )

    return sk.database.MieDatabase(dist, refrac, np.arange(1300, 1401.0, 5.0))


def process_l1b_to_l2_file(l1b_file: Path, output_folder: Path, cal_db_path: Path):
    # Load in the data
    cal_db = xr.open_dataset(cal_db_path)
    l1b_data = L1bDataSet(l1b_file)
    por_data = xr.open_dataset(l1b_data.l2_por_path)

    # And get the output file name
    out_file = output_folder.joinpath(l1b_data.l2_path.stem + ".nc")

    if not out_file.exists():
        logging.info("Processing %s to %s", l1b_file.stem, out_file.stem)

        writer = L2FileWriter(process_l1b_to_l2(l1b_data, por_data, cal_db))

        writer.save(out_file)


def process_l1b_to_l2(l1b_data: L1bDataSet, por_data: xr.Dataset, cal_db: Path):
    low_alt = 12000

    l2s = []
    for image in range(len(l1b_data.ds.time)):
        l1b_image = l1b_data.image(
            image, low_alt=low_alt, high_alt=32000, row_reduction=1
        )
        por_image = por_data.isel(time=image)

        good = ~np.isnan(por_image.pressure.to_numpy())
        anc = Ancillary(
            por_image.altitude.to_numpy()[good],
            por_image.pressure.to_numpy()[good],
            por_image.temperature.to_numpy()[good],
        )

        ret = Retrieval(
            observation=l1b_image,
            forward_model_cfg={
                "meas": {
                    "kwargs": {
                        "lineshape_fn": lambda w: UserLineShape(
                            cal_db.hires_wavenumber.to_numpy(),
                            cal_db.sel(sample_wavenumber=1e7 / w, method="nearest")[
                                "ils"
                            ].to_numpy(),
                            False,
                            integration_fraction=0.95,
                        ),
                        "spectral_native_coordinate": "wavenumber_cminv",
                    },
                    "class": SHOWForwardModel,
                },
            },
            minimizer="scipy",
            ancillary=anc,
            l1_kwargs={},
            model_kwargs={"num_threads": 8},
            minimizer_kwargs={"max_nfev": 20},
            target_kwargs={},
            measvec={
                "meas": MeasurementVector(
                    lambda l1, ctxt, **kwargs: select(  # noqa: ARG005
                        l1, wavenumber=slice(7312, 7340), **kwargs
                    )
                )
            },
            state_kwargs={
                "absorbers": {
                    "h2o": {
                        "prior_influence": 1e2,
                        "tikh_factor": 2.5e4,
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
                                "prior_influence": 1e-4,
                                "tikh_factor": 1e-1,
                                "min_value": 0,
                                "max_value": 1e-3,
                            },
                        },
                        "prior": {
                            "extinction_per_m": {"type": "testing"},
                        },
                    },
                },
                "altitude_grid": np.arange(0.0, 65001.0, 1000.0),
            },
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
            process_l1b_to_l2(
                file,
                Path(
                    r"/Users/dannyz/OneDrive - University of Saskatchewan/SHOW/er2_2023/sci_flight/l2",
                ),
                Path(r"/Users/dannyz/Documents/sds/ils.nc"),
            )
