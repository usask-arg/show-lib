from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

NUM_TIME = 2
NUM_LOS = 3
NUM_COLUMNS = 8
NUM_SAMPLES = 4
LITTROW = 7330.0
SAMPLE_SPACING = 0.5


def _var(dims, values, units=None):
    attrs = {} if units is None else {"units": units}
    return xr.Variable(dims, values, attrs=attrs)


def _times():
    return np.array(
        ["2026-11-03T12:00:00", "2026-11-03T12:00:01"], dtype="datetime64[ns]"
    )


def _geometry(prefix_dims):
    """Geometry variables shared by L1A and L1B."""
    shape = (NUM_TIME, NUM_LOS)
    return {
        "tangent_altitude": _var(
            prefix_dims, np.broadcast_to([10e3, 20e3, 30e3], shape).copy(), "m"
        ),
        "tangent_latitude": _var(prefix_dims, np.full(shape, 52.0), "degrees_north"),
        "tangent_longitude": _var(prefix_dims, np.full(shape, -106.0), "degrees_east"),
        "spacecraft_latitude": _var(["time"], np.full(NUM_TIME, 50.0), "degrees_north"),
        "spacecraft_longitude": _var(
            ["time"], np.full(NUM_TIME, -106.0), "degrees_east"
        ),
        "spacecraft_altitude": _var(["time"], np.full(NUM_TIME, 500e3), "m"),
        "solar_zenith_angle": _var(prefix_dims, np.full(shape, 60.0), "degree"),
        "relative_solar_azimuth_angle": _var(
            prefix_dims, np.full(shape, 30.0), "degree"
        ),
        "los_azimuth_angle": _var(prefix_dims, np.full(shape, 90.0), "degree"),
    }


@pytest.fixture
def l1a() -> xr.Dataset:
    image_dims = ["time", "los", "pixelcolumn"]
    data_vars = {
        "image": _var(
            image_dims, np.full((NUM_TIME, NUM_LOS, NUM_COLUMNS), 1000.0), "count"
        ),
        "exposure_time": _var(["time"], np.full(NUM_TIME, 0.1), "s"),
        "num_averaged_exposures": _var(["time"], np.full(NUM_TIME, 2), "1"),
        "saturated": xr.Variable(
            image_dims, np.zeros((NUM_TIME, NUM_LOS, NUM_COLUMNS), dtype=bool)
        ),
        **_geometry(["time", "los"]),
    }
    return xr.Dataset(
        data_vars,
        coords={"time": _times()},
        attrs={"schema_version": "0.2.0"},
    )


@pytest.fixture
def l1a_simulation() -> xr.Dataset:
    return xr.Dataset(
        {
            "image_noiseless": _var(
                ["time", "los", "pixelcolumn"],
                np.full((NUM_TIME, NUM_LOS, NUM_COLUMNS), 1000.0),
                "count",
            ),
            "noise_enabled": xr.Variable([], True),
            "noise_seed": xr.Variable([], np.int64(7)),
        }
    )


@pytest.fixture
def l1b() -> xr.Dataset:
    spectral_dims = ["time", "los", "spectral_sample"]
    shape = (NUM_TIME, NUM_LOS, NUM_SAMPLES)
    wavenumber = LITTROW + SAMPLE_SPACING * np.arange(1, NUM_SAMPLES + 1)
    data_vars = {
        "radiance": _var(spectral_dims, np.full(shape, 1e-3), "W m-2 sr-1 nm-1"),
        "radiance_noise": _var(spectral_dims, np.full(shape, 1e-5), "W m-2 sr-1 nm-1"),
        "saturated_fraction": _var(["time", "los"], np.zeros((NUM_TIME, NUM_LOS)), "1"),
        **_geometry(["time", "los"]),
    }
    return xr.Dataset(
        data_vars,
        coords={
            "time": _times(),
            "los": _var(["los"], np.arange(NUM_LOS), "1"),
            "wavenumber": _var(
                ["los", "spectral_sample"],
                np.broadcast_to(wavenumber, (NUM_LOS, NUM_SAMPLES)).copy(),
                "cm-1",
            ),
        },
        attrs={
            "schema_version": "0.2.0",
            "apodization_window": "kaiser",
            "apodization_beta": 10.0,
            "radiance_noise_model": "detector",
        },
    )


@pytest.fixture
def cal_db() -> xr.Dataset:
    opd_edges = np.linspace(-0.4, 0.4, NUM_COLUMNS + 1)
    sample_wavenumber = LITTROW + SAMPLE_SPACING * np.arange(1, NUM_SAMPLES + 1)
    transmission_wavenumber = np.linspace(LITTROW - 5.0, LITTROW + 10.0, 31)
    data_vars = {
        "opd": _var(
            ["los_edge", "pixelcolumn_edge"],
            np.broadcast_to(opd_edges, (NUM_LOS + 1, NUM_COLUMNS + 1)).copy(),
            "cm",
        ),
        "littrow_wavenumber": _var(["los_edge"], np.full(NUM_LOS + 1, LITTROW), "cm-1"),
        "pixel_etendue": _var(
            ["los", "pixelcolumn"], np.full((NUM_LOS, NUM_COLUMNS), 1e-10), "m2 sr"
        ),
        "spectral_transmission": _var(
            ["transmission_wavenumber"],
            np.full(transmission_wavenumber.shape, 0.8),
            "1",
        ),
        "detector_gain": _var([], 2.0, "1"),
        "detector_read_noise": _var([], 10.0, "1"),
        "sample_wavenumber": _var(
            ["los", "spectral_sample"],
            np.broadcast_to(sample_wavenumber, (NUM_LOS, NUM_SAMPLES)).copy(),
            "cm-1",
        ),
    }
    return xr.Dataset(
        data_vars,
        coords={
            "transmission_wavenumber": _var(
                ["transmission_wavenumber"], transmission_wavenumber, "cm-1"
            )
        },
        attrs={"schema_version": "0.3.0", "valid_from": "2026-11-03T00:00:00Z"},
    )
