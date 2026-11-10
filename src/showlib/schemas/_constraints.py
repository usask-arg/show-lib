"""Checks for the `constraints` section of each schema, keyed by constraint id.

Each check takes the dataset and an optional calibration database and returns a
list of problems. A check returns nothing when a field it needs is missing; the
per-field checks already report missing fields.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import xarray as xr

# Tolerances for the dispersion-reference constraint
_SCALE_ATOL = 1e-6
_PHASE_ATOL_RAD = 1e-6
_PHASE_SLOPE_ATOL_RAD_PER_CMINV = 1e-6

Check = Callable[[xr.Dataset, xr.Dataset | None], list[str]]


def _has(ds: xr.Dataset, *names: str) -> bool:
    return all(name in ds.variables for name in names)


def _bad_rows(rows: np.ndarray) -> str:
    return ", ".join(str(row) for row in np.flatnonzero(rows))


# ------------------------------------------------------------------ L1A / L1B
def _time_increasing(ds, _cal_db):
    if not _has(ds, "time") or ds["time"].ndim != 1:
        return []
    if np.all(np.diff(ds["time"].to_numpy()) > np.timedelta64(0, "ns")):
        return []
    return ["time is not strictly increasing"]


def _all_finite(*names: str) -> Check:
    def check(ds, _cal_db):
        return [
            f"{name} has {np.count_nonzero(~np.isfinite(ds[name]))} non-finite values"
            for name in names
            if _has(ds, name) and not np.all(np.isfinite(ds[name]))
        ]

    return check


def _detector_matches_calibration(ds, cal_db):
    if cal_db is None:
        return []
    return [
        f"{dim} has length {ds.sizes[dim]}, but the calibration database has "
        f"{cal_db.sizes[dim]}"
        for dim in ("los", "pixelcolumn")
        if dim in ds.sizes
        and dim in cal_db.sizes
        and ds.sizes[dim] != cal_db.sizes[dim]
    ]


def _los_rows(ds, _cal_db):
    if not _has(ds, "los") or ds["los"].ndim != 1:
        return []
    if np.all(np.diff(ds["los"].to_numpy()) > 0):
        return []
    return ["los is not strictly increasing"]


def _wavenumber_increasing(ds, _cal_db):
    if not _has(ds, "wavenumber"):
        return []
    if np.all(ds["wavenumber"].diff("spectral_sample") > 0):
        return []
    return ["wavenumber is not strictly increasing along spectral_sample"]


# --------------------------------------------------------------------- CAL_DB
def _edge_lengths(ds, _cal_db):
    return [
        f"{edge} has length {ds.sizes[edge]}, expected {ds.sizes[dim] + 1}"
        for dim, edge in (("los", "los_edge"), ("pixelcolumn", "pixelcolumn_edge"))
        if dim in ds.sizes and edge in ds.sizes and ds.sizes[edge] != ds.sizes[dim] + 1
    ]


def _opd_monotonic(ds, _cal_db):
    if not _has(ds, "opd"):
        return []
    opd = ds["opd"].transpose("los_edge", "pixelcolumn_edge").to_numpy()
    if not np.all(np.isfinite(opd)):
        return ["opd has non-finite values"]
    steps = np.diff(opd, axis=1)
    monotonic = np.all(steps > 0, axis=1) | np.all(steps < 0, axis=1)
    if np.all(monotonic):
        return []
    return [f"opd is not strictly monotonic at row edges {_bad_rows(~monotonic)}"]


def _transmission_covers_samples(ds, _cal_db):
    if not _has(
        ds, "transmission_wavenumber", "spectral_transmission", "sample_wavenumber"
    ):
        return []
    grid = ds["transmission_wavenumber"].to_numpy()
    order = np.argsort(grid)
    samples = ds["sample_wavenumber"].to_numpy().ravel()

    problems = []
    if grid.min() > samples.min() or grid.max() < samples.max():
        problems.append(
            f"transmission_wavenumber spans [{grid.min()}, {grid.max()}], but samples "
            f"span [{samples.min()}, {samples.max()}]"
        )
    transmission = np.interp(
        samples, grid[order], ds["spectral_transmission"].to_numpy()[order]
    )
    if np.any(transmission <= 0):
        problems.append("spectral_transmission is not positive at every sample")
    return problems


def _samples_increasing(ds, _cal_db):
    if not _has(ds, "sample_wavenumber"):
        return []
    if np.all(ds["sample_wavenumber"].diff("spectral_sample") > 0):
        return []
    return ["sample_wavenumber is not strictly increasing along spectral_sample"]


def _row_offsets(ds: xr.Dataset) -> np.ndarray | None:
    """sample_wavenumber minus each row's mean Littrow wavenumber, or None."""
    if not _has(ds, "sample_wavenumber", "littrow_wavenumber"):
        return None
    samples = ds["sample_wavenumber"].transpose("los", "spectral_sample").to_numpy()
    edges = ds["littrow_wavenumber"].to_numpy()
    if len(edges) != len(samples) + 1:
        return None
    littrow = 0.5 * (edges[:-1] + edges[1:])
    return samples - littrow[:, np.newaxis]


def _samples_off_dc(ds, _cal_db):
    offsets = _row_offsets(ds)
    if offsets is None:
        return []
    distance = np.min(np.abs(offsets), axis=1)
    if offsets.shape[1] > 1:
        half_spacing = 0.5 * np.min(np.abs(np.diff(offsets, axis=1)), axis=1)
        too_close = distance < half_spacing
    else:
        too_close = np.isclose(distance, 0)
    if not np.any(too_close):
        return []
    return [f"a sample is at the Littrow wavenumber in rows {_bad_rows(too_close)}"]


def _samples_one_sided(ds, _cal_db):
    offsets = _row_offsets(ds)
    if offsets is None:
        return []
    one_sided = np.all(offsets > 0, axis=1) | np.all(offsets < 0, axis=1)
    if np.all(one_sided):
        return []
    rows = _bad_rows(~one_sided)
    return [f"samples lie on both sides of the Littrow wavenumber in rows {rows}"]


_DISPERSION_FUNCTIONS = ("fringe_frequency_scale", "dispersion_phase")
_DISPERSION_GRID = ("dispersion_wavenumber", "dispersion_reference_wavenumber")


def _dispersion_complete(ds, _cal_db):
    if not any(_has(ds, name) for name in _DISPERSION_FUNCTIONS):
        return []
    missing = [name for name in _DISPERSION_GRID if not _has(ds, name)]
    if not missing:
        return []
    return [f"dispersion functions are present but {', '.join(missing)} missing"]


def _dispersion_covers_band(ds, _cal_db):
    if not _has(ds, "dispersion_wavenumber", "sample_wavenumber"):
        return []
    band = ds["sample_wavenumber"].to_numpy().ravel()
    if _has(ds, "transmission_wavenumber", "spectral_transmission"):
        passband = ds["spectral_transmission"].to_numpy() > 0
        band = np.concatenate(
            [band, ds["transmission_wavenumber"].to_numpy()[passband]]
        )
    grid = ds["dispersion_wavenumber"].to_numpy()
    if grid.min() <= band.min() and grid.max() >= band.max():
        return []
    message = (
        f"dispersion_wavenumber spans [{grid.min()}, {grid.max()}], but the band "
        f"spans [{band.min()}, {band.max()}]"
    )
    return [message]


def _dispersion_reference(ds, _cal_db):
    if not _has(ds, *_DISPERSION_GRID):
        return []
    grid = ds["dispersion_wavenumber"].to_numpy()
    if len(grid) < 2:
        return ["dispersion_wavenumber needs at least two points to define a slope"]
    order = np.argsort(grid)
    grid = grid[order]
    reference = float(ds["dispersion_reference_wavenumber"])

    def at_reference(values):
        return float(np.interp(reference, grid, values))

    problems = []
    if _has(ds, "fringe_frequency_scale"):
        scale = ds["fringe_frequency_scale"].to_numpy()[order]
        if abs(at_reference(scale) - 1) > _SCALE_ATOL:
            problems.append(
                f"fringe_frequency_scale is {at_reference(scale)} at the reference, "
                "expected 1"
            )
    if _has(ds, "dispersion_phase"):
        phase = ds["dispersion_phase"].to_numpy()[order]
        if abs(at_reference(phase)) > _PHASE_ATOL_RAD:
            problems.append(
                f"dispersion_phase is {at_reference(phase)} at the reference, "
                "expected 0"
            )
        slope = at_reference(np.gradient(phase, grid))
        if abs(slope) > _PHASE_SLOPE_ATOL_RAD_PER_CMINV:
            problems.append(
                f"dispersion_phase has slope {slope} at the reference, expected 0"
            )
    return problems


CONSTRAINT_CHECKS: dict[str, dict[str, Check]] = {
    "l1a": {
        "time-increasing": _time_increasing,
        "image-finite": _all_finite("image"),
        "detector-matches-calibration": _detector_matches_calibration,
    },
    "l1b": {
        "time-increasing": _time_increasing,
        "los-rows": _los_rows,
        "wavenumber-increasing": _wavenumber_increasing,
        "finite-values": _all_finite("radiance", "radiance_noise"),
    },
    "cal_db": {
        "edge-lengths": _edge_lengths,
        "opd-monotonic": _opd_monotonic,
        "transmission-covers-samples": _transmission_covers_samples,
        "samples-increasing": _samples_increasing,
        "samples-off-dc": _samples_off_dc,
        "samples-one-sided": _samples_one_sided,
        "dispersion-complete": _dispersion_complete,
        "dispersion-covers-band": _dispersion_covers_band,
        "dispersion-reference": _dispersion_reference,
    },
}

DEFERRED_CONSTRAINTS: dict[str, dict[str, str]] = {
    "l1b": {
        "uniform-processing": (
            "Enforced by the granule writer; a file holds one set of global attributes."
        ),
    },
    "cal_db": {
        "invertible": "Needs the forward model, which is not implemented yet.",
    },
}
