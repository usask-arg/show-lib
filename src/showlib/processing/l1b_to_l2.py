from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import sasktran2 as sk
import xarray as xr
from skretrieval.core.lineshape import UserLineShape
from skretrieval.retrieval.processing import Retrieval
from skretrieval.util import configure_log

from showlib.l1b.data import L1bDataSet
from showlib.l2.ancillary import Ancillary
from showlib.l2.data import L2FileWriter, L2Profile
from showlib.l2.optical import h2o_optical_property
from showlib.l2.preprocessor import l2_preprocessor
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


def process_l1b_to_l2_file(
    l1b_file: Path, output_folder: Path, cal_db_path: Path, **kwargs
):
    # Load in the data
    cal_db = xr.open_dataset(cal_db_path)
    l1b_data = L1bDataSet.from_file(l1b_file)
    por_data = xr.open_dataset(l1b_data.l2_por_path)

    if "hires_wavenumber" not in cal_db:
        # OLD version of the CAL_DB for ER2
        cal_db = cal_db.rename({"wavenumbers": "hires_wavenumber"})

    # And get the output file name
    out_file = output_folder.joinpath(l1b_data.l2_path.stem + ".nc")

    if not out_file.exists():
        logging.info("Processing %s to %s", l1b_file.stem, out_file.stem)

        writer = L2FileWriter(process_l1b_to_l2(l1b_data, por_data, cal_db, **kwargs))

        writer.save(out_file)


def process_l1b_to_l2(
    l1b_data: L1bDataSet,
    por_data: xr.Dataset,
    cal_db: xr.Dataset,
    process_slice=None,
    **kwargs,
):
    if "altitude_grid" not in kwargs:
        altitude_grid = np.unique(
            np.concatenate(
                (
                    np.arange(0, 9000.0, 1000.0),
                    np.arange(9000, 35000, 500.0),
                    np.arange(35000, 40000, 1000),
                    np.arange(40000, 65001, 5000),
                )
            )
        )
    else:
        altitude_grid = kwargs["altitude_grid"]
    if process_slice is None:
        process_slice = range(len(l1b_data.ds.time))

    preprocessor_data = l2_preprocessor(
        l1b_data, por_data, cal_db, process_slice, altitude_grid=altitude_grid, **kwargs
    )

    upper_bound = kwargs.get("upper_bound", 30000)

    l2s = []
    for i, image in enumerate(process_slice):
        preprocessor = preprocessor_data[i]

        l1b_image = l1b_data.image(
            image,
            low_alt=preprocessor["low_alt"],
            high_alt=upper_bound,
            row_reduction=kwargs.get("row_reduction", 1),
            low_wvnumber_filter=kwargs.get("low_wvnumber_filter", 0),
            high_wvnumber_filter=kwargs.get("high_wvnumber_filter", 1e10),
            override_los=kwargs.get("override_los", False),
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
                        "model_res_cminv": kwargs.get("model_res_cminv", 0.01),
                        "round_decimal": 3,
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
            state_kwargs={
                "absorbers": {
                    "h2o": {
                        "prior_influence": 1e0,
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
                        "initial_guess": preprocessor["aerosol"],
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
                "surface": {
                    "lambertian_albedo": {
                        "prior_influence": 0,
                        "tikh_factor": 1e6,
                        "log_space": False,
                        "wavelengths": np.array([1e7 / max_wv, 1e7 / min_wv]),
                        "initial_value": 0.3,
                        "out_bounds_mode": "extend",
                    },
                },
                "shifts": {
                    "wavenumber": {
                        "type": "wavenumber_shift",
                        "num_los": len(l1b_image.skretrieval_l1()["meas"].data.los),
                    }
                },
                "splines": {
                    "constant_los": {
                        "low_wavelength_nm": 1e7 / max_wv,
                        "high_wavelength_nm": 1e7 / min_wv,
                        "num_wv": 2,
                        "order": 1,
                    }
                },
                "altitude_grid": altitude_grid,
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
