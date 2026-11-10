# ruff: noqa: T201  (the report is printed)
"""End-to-end self-consistency run: SASKTRAN2 radiance -> L1A -> L1B -> L2.

L1A is generated with showlib's own forward model A (CalibrationDatabase), and
the retrieval models the measurement with the same A, so this is an "inverse
crime" test. It checks that the pipeline runs and that the retrieval recovers the
truth when the instrument model is perfect. It does not test instrument-model
fidelity; that needs an independent simulator.

The truth atmosphere mirrors the retrieval's forward model: same altitude grid,
Ancillary (Rayleigh, US76 pressure and temperature, solar irradiance, MIPAS CH4),
H2O and aerosol optical properties, Lambertian surface, tangent-altitude geometry,
and engine settings.

Usage:
    uv run python workflows/self_consistency/run.py OUTPUT_DIR [--noise] [--threads N]
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import sasktran2 as sk
import xarray as xr
import yaml
from skretrieval.geodetic import geodetic

from showlib.cal_db import CalibrationDatabase
from showlib.cal_db.ideal import ideal_calibration_database
from showlib.l1a.data import L1ADataSet
from showlib.l1a.simulate import simulate_l1a
from showlib.l2.ancillary import Ancillary
from showlib.l2.config import load_retrieval_config
from showlib.l2.lineshape import SHOWLineShapes
from showlib.l2.optical import h2o_optical_property
from showlib.processing.l1a_to_l1b import process_l1a_to_l1b
from showlib.processing.l1b_to_l2 import (
    process_l1b_to_l2_file,
    stratospheric_aerosol_optical_property,
)

NAME = "20261110T1200.v0_0_1.SIM"
TIME = np.datetime64("2026-11-10T12:00:00", "ns")
LATITUDE, LONGITUDE = 52.0, -106.0
SPACECRAFT_ALTITUDE_M = 500e3
SOLAR_ZENITH_ANGLE = 60.0
RELATIVE_SOLAR_AZIMUTH = 30.0
TANGENT_ALTITUDES_M = np.arange(5e3, 35e3 + 1, 0.5e3)
AUTO_EXPOSURE_FULL_WELL_FRACTION = 0.8
SURFACE_ALBEDO = 0.3
TROPOPAUSE_ALTITUDE_M = 11e3

# The default retrieval altitude grid in showlib.processing.l1b_to_l2
ALTITUDE_GRID_M = np.unique(
    np.concatenate(
        (
            np.arange(0, 9000.0, 1000.0),
            np.arange(9000, 35000, 500.0),
            np.arange(35000, 40000, 1000),
            np.arange(40000, 65001, 5000),
        )
    )
)


def truth_h2o_vmr(altitude_m: np.ndarray) -> np.ndarray:
    """Moist troposphere, ~4.5 ppm stratosphere, and a 1.5 ppm bump at 18 km."""
    z = altitude_m / 1e3
    return 4.5e-6 + 1e-2 * np.exp(-z / 2.2) + 1.5e-6 * np.exp(-(((z - 18) / 3) ** 2))


def us76_pressure_temperature(altitude_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    geometry = sk.Geometry1D(
        0.5,
        0.0,
        6372e3,
        altitude_m,
        sk.InterpolationMethod.LinearInterpolation,
        sk.GeometryType.Spherical,
    )
    atmo = sk.Atmosphere(geometry, sk.Config(), wavelengths_nm=np.array([1364.0]))
    sk.climatology.us76.add_us76_standard_atmosphere(atmo)
    return atmo.pressure_pa.copy(), atmo.temperature_k.copy()


def engine_config(num_threads: int) -> sk.Config:
    """Same settings the L1B->L2 retrieval passes as model_kwargs."""
    config = sk.Config()
    config.num_threads = num_threads
    config.los_refraction = True
    config.multiple_scatter_source = sk.MultipleScatterSource.TwoStream
    config.num_streams = 2
    config.wavelength_batch_size = 128
    return config


def simulate_radiance(model_wavenumber: np.ndarray, num_threads: int) -> np.ndarray:
    """Truth radiance on the model grid, shape (num_model, num_rows)."""
    wavelength_nm = 1e7 / model_wavenumber[::-1]
    pressure, temperature = us76_pressure_temperature(ALTITUDE_GRID_M)

    geo = geodetic()
    geo.from_lat_lon_alt(LATITUDE, LONGITUDE, 0.0)
    geometry = sk.Geometry1D(
        np.cos(np.deg2rad(SOLAR_ZENITH_ANGLE)),
        0.0,
        np.linalg.norm(geo.location),
        ALTITUDE_GRID_M,
        sk.InterpolationMethod.LinearInterpolation,
        sk.GeometryType.Spherical,
    )
    viewing = sk.ViewingGeometry()
    for tangent_altitude in TANGENT_ALTITUDES_M:
        viewing.add_ray(
            sk.TangentAltitudeSolar(
                tangent_altitude,
                np.deg2rad(RELATIVE_SOLAR_AZIMUTH),
                SPACECRAFT_ALTITUDE_M,
                np.cos(np.deg2rad(SOLAR_ZENITH_ANGLE)),
            )
        )

    config = engine_config(num_threads)
    atmo = sk.Atmosphere(geometry, config, wavelengths_nm=wavelength_nm)
    Ancillary(ALTITUDE_GRID_M, pressure, temperature).add_to_atmosphere(atmo)
    atmo["h2o"] = sk.constituent.VMRAltitudeAbsorber(
        h2o_optical_property(), ALTITUDE_GRID_M, truth_h2o_vmr(ALTITUDE_GRID_M)
    )
    aerosol = sk.test_util.scenarios.test_aerosol_constituent(ALTITUDE_GRID_M)
    atmo["stratospheric_aerosol"] = sk.constituent.ExtinctionScatterer(
        stratospheric_aerosol_optical_property(),
        ALTITUDE_GRID_M,
        np.clip(aerosol.extinction_per_m, 1e-15, None),
        1350,
        "extend",
    )
    atmo["surface"] = sk.constituent.LambertianSurface(
        np.full(2, SURFACE_ALBEDO),
        np.array([wavelength_nm.min(), wavelength_nm.max()]),
        "extend",
    )
    geometry.refractive_index = sk.optical.refraction.ciddor_index_of_refraction(
        atmo.temperature_k,
        atmo.pressure_pa,
        np.zeros_like(atmo.temperature_k),
        450,
        np.nanmean(wavelength_nm),
    )

    radiance = sk.Engine(config, geometry, viewing).calculate_radiance(atmo)
    # Model wavelengths increase, so reverse to increasing wavenumber
    return radiance["radiance"].isel(stokes=0).to_numpy()[::-1]


def simulate_granule(
    cal_db: CalibrationDatabase,
    model_wavenumber: np.ndarray,
    radiance: np.ndarray,
    noise_seed: int | None,
) -> L1ADataSet:
    """L1A for one image from radiance on the model grid, shape (num_model, num_rows)."""
    num_rows = cal_db.num_rows
    geometry = {
        "time": TIME,
        "tangent_altitude": TANGENT_ALTITUDES_M,
        "tangent_latitude": np.full(num_rows, LATITUDE),
        "tangent_longitude": np.full(num_rows, LONGITUDE),
        "spacecraft_latitude": LATITUDE,
        "spacecraft_longitude": LONGITUDE,
        "spacecraft_altitude": SPACECRAFT_ALTITUDE_M,
        "solar_zenith_angle": np.full(num_rows, SOLAR_ZENITH_ANGLE),
        "relative_solar_azimuth_angle": np.full(num_rows, RELATIVE_SOLAR_AZIMUTH),
        "los_azimuth_angle": np.full(num_rows, 90.0),
    }
    return simulate_l1a(
        cal_db,
        model_wavenumber,
        radiance.T[np.newaxis],
        [geometry],
        full_well_fraction=AUTO_EXPOSURE_FULL_WELL_FRACTION,
        noise=noise_seed is not None,
        seed=noise_seed,
    )


def program_of_record() -> xr.Dataset:
    pressure, temperature = us76_pressure_temperature(ALTITUDE_GRID_M)
    return xr.Dataset(
        {
            "pressure": (["time", "altitude"], pressure[np.newaxis]),
            "temperature": (["time", "altitude"], temperature[np.newaxis]),
            "tropopause_altitude": (["time"], [TROPOPAUSE_ALTITUDE_M]),
        },
        coords={"altitude": ALTITUDE_GRID_M, "time": [TIME]},
    )


def main(
    output_dir: Path, noise_seed: int | None, num_threads: int, config: dict
) -> None:
    for folder in ["l1a", "l1b", "por", "l2"]:
        output_dir.joinpath(folder).mkdir(parents=True, exist_ok=True)

    timings = {}

    def timed(label, fn, *args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        timings[label] = time.perf_counter() - start
        logging.info("%s: %.1f s", label, timings[label])
        return result

    cal_db = timed(
        "calibration database", ideal_calibration_database, len(TANGENT_ALTITUDES_M)
    )
    cal_db_path = output_dir / "cal_db.nc"
    cal_db.ds.to_netcdf(cal_db_path)
    model_wavenumber = SHOWLineShapes(cal_db, apodization=None).model_wavenumber
    logging.info(
        "%d rows, %d columns, %d samples, %d model wavenumbers",
        cal_db.num_rows,
        cal_db.num_columns,
        cal_db.ds.sizes["spectral_sample"],
        len(model_wavenumber),
    )

    radiance = timed(
        "SASKTRAN2 truth radiance", simulate_radiance, model_wavenumber, num_threads
    )
    l1a = timed("L1A", simulate_granule, cal_db, model_wavenumber, radiance, noise_seed)
    l1a.save(output_dir / "l1a" / f"HAWC_H2OL_Counts_L1A_{NAME}.nc")
    image = l1a.ds["image"].to_numpy()
    logging.info(
        "L1A counts: %.3g to %.3g, exposure %.3g s",
        image.min(),
        image.max(),
        float(l1a.ds["exposure_time"].isel(time=0)),
    )

    l1b = timed("L1A -> L1B", process_l1a_to_l1b, l1a, cal_db)
    l1b_path = output_dir / "l1b" / f"HAWC_H2OL_Radiances_L1B_{NAME}.nc"
    l1b.save(l1b_path)
    program_of_record().to_netcdf(
        output_dir / "por" / f"HAWC_H2OL_Wvapor_L2_POR_{NAME}.nc"
    )

    timed(
        "L1B -> L2",
        process_l1b_to_l2_file,
        l1b_path,
        output_dir / "l2",
        cal_db_path,
        config,
    )

    with xr.open_dataset(output_dir / "l2" / f"HAWC_H2OL_Wvapor_L2_{NAME}.nc") as l2:
        report(l2.load())
    for label, seconds in timings.items():
        print(f"{label:>28}: {seconds:8.1f} s")


def report(l2: xr.Dataset) -> None:
    profile = l2.isel(time=0) if "time" in l2.dims else l2
    altitude = profile["altitude"].to_numpy()
    retrieved = profile["h2o_vmr"].to_numpy()
    truth = truth_h2o_vmr(altitude)
    error = (
        profile["h2o_vmr_1sigma"].to_numpy() if "h2o_vmr_1sigma" in profile else None
    )
    prior = profile["h2o_vmr_prior"].to_numpy()
    print(
        f"{'altitude km':>11} {'truth ppm':>10} {'prior':>8} {'retrieved':>10} "
        f"{'diff %':>8} {'1sig %':>6}"
    )
    for i in np.flatnonzero((altitude >= 8e3) & (altitude <= 30e3)):
        sigma = "" if error is None else f"{100 * error[i] / truth[i]:6.1f}"
        print(
            f"{altitude[i] / 1e3:11.1f} {1e6 * truth[i]:10.2f} {1e6 * prior[i]:8.2f} "
            f"{1e6 * retrieved[i]:10.2f} "
            f"{100 * (retrieved[i] / truth[i] - 1):8.1f} {sigma}"
        )


def _merge_overrides(base: dict, extra: dict) -> dict:
    """Command-line settings layered over a --config file."""
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_overrides(merged[key], value)
        else:
            merged[key] = value
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--noise", action="store_true", help="add shot and read noise")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--preprocessor-wvnum",
        type=float,
        default=7335.165,
        help=(
            "sample wavenumber [cm-1] for the aerosol preprocessor; the default is the "
            "least H2O-absorbed sample of the ideal calibration database"
        ),
    )
    parser.add_argument(
        "--tangent-spacing-km",
        type=float,
        default=0.5,
        help="spacing of the detector rows' tangent altitudes",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML retrieval configuration merged into the defaults",
    )
    args = parser.parse_args()
    TANGENT_ALTITUDES_M = np.arange(5e3, 35e3 + 1, args.tangent_spacing_km * 1e3)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main(
        args.output_dir,
        args.seed if args.noise else None,
        args.threads,
        load_retrieval_config(
            _merge_overrides(
                yaml.safe_load(args.config.read_text()) if args.config else {},
                {
                    "preprocessor": {"wavenumber_cminv": args.preprocessor_wvnum},
                    "engine": {"num_threads": args.threads},
                },
            )
        ),
    )
