from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from showlib.schemas import apply_metadata


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        help="run end-to-end tests that execute SASKTRAN2 retrievals",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="slow; run with --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


def _cal_db(
    num_rows: int = 2,
    num_columns: int = 16,
    opd_spacing: float = 0.1,
    littrow: float = 7330.0,
    sample_indices=(-3, -2, -1),
    opd: np.ndarray | None = None,
    littrow_edges: np.ndarray | None = None,
    fringe_phase: np.ndarray | None = None,
    fringe_visibility: np.ndarray | None = None,
    pixel_etendue: float | np.ndarray = 1.5e-10,
    transmission: float = 0.8,
    detector_gain: float = 2.0,
    detector_read_noise: float = 10.0,
) -> xr.Dataset:
    """Calibration database for a centred, uniform OPD grid unless overridden.

    Samples sit on the natural grid at littrow + index / (num_columns · opd_spacing).
    """
    if opd is None:
        edges = (np.arange(num_columns + 1) - num_columns / 2) * opd_spacing
        opd = np.broadcast_to(edges, (num_rows + 1, num_columns + 1)).copy()
    if littrow_edges is None:
        littrow_edges = np.full(num_rows + 1, littrow)
    natural_spacing = 1 / (num_columns * opd_spacing)
    samples = littrow + np.asarray(sample_indices) * natural_spacing
    transmission_wavenumber = np.linspace(samples.min() - 5, samples.max() + 5, 101)

    data_vars = {
        "opd": (["los_edge", "pixelcolumn_edge"], opd),
        "littrow_wavenumber": (["los_edge"], littrow_edges),
        "pixel_etendue": (
            ["los", "pixelcolumn"],
            np.broadcast_to(pixel_etendue, (num_rows, num_columns)).copy(),
        ),
        "spectral_transmission": (
            ["transmission_wavenumber"],
            np.full(transmission_wavenumber.shape, transmission),
        ),
        "detector_gain": ((), detector_gain),
        "detector_read_noise": ((), detector_read_noise),
        "sample_wavenumber": (
            ["los", "spectral_sample"],
            np.broadcast_to(samples, (num_rows, len(samples))).copy(),
        ),
    }
    if fringe_phase is not None:
        data_vars["fringe_phase"] = (["los_edge", "pixelcolumn_edge"], fringe_phase)
    if fringe_visibility is not None:
        data_vars["fringe_visibility"] = (["los", "pixelcolumn"], fringe_visibility)

    ds = xr.Dataset(
        data_vars,
        coords={"transmission_wavenumber": transmission_wavenumber},
        attrs={"valid_from": "2026-11-03T00:00:00Z"},
    )
    return apply_metadata(ds, "cal_db")


@pytest.fixture
def make_cal_db():
    """Factory for calibration database datasets; see _cal_db for parameters."""
    return _cal_db
