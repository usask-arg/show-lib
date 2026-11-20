"""SHOW calibration database and the forward model it parameterizes.

The model is defined in the `model` section of src/showlib/schemas/cal_db.yaml.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import xarray as xr

from showlib.constant import PLANCK_CONSTANT_J_S, SPEED_OF_LIGHT_M_S
from showlib.schemas import check

# Samples per fringe period in the default model_wavenumber grid
_FRINGE_SAMPLES_PER_PERIOD = 20
_TABLE_SAMPLES_PER_STEP = 4
# Gauss-Legendre nodes along the row direction when the phase does not change
_MIN_ROW_NODES = 4


def bin_width(wavenumber) -> np.ndarray:
    """Trapezoid-rule weights [cm-1] for radiance tabulated on ``wavenumber``.

    ``bin_width(grid) @ values`` integrates piecewise-linear values exactly; use
    it for the ``bin_width`` of forward_operator and radiance_weight.
    """
    wavenumber = np.asarray(wavenumber, dtype=float)
    widths = np.gradient(wavenumber)
    widths[0] = 0.5 * (wavenumber[1] - wavenumber[0])
    widths[-1] = 0.5 * (wavenumber[-1] - wavenumber[-2])
    return widths


def pixel_averaged_fringe(
    ds: xr.Dataset, row: int, wavenumber, num_row_nodes: int | None = None
) -> np.ndarray:
    """V(y,x) · C(y,x,nu) for a calibration dataset: the fringe averaged over each
    pixel of a row, shape (num_columns, len(wavenumber)).

    At any position v along the row direction, the fringe phase is linear across
    the pixel, so the average across the pixel is exact: cos(mid-phase) · sinc. The
    average along the row direction uses Gauss-Legendre quadrature with
    ``num_row_nodes`` nodes. By default the number of nodes follows the largest
    phase change between the row's two edges (for example from a grating tilt):
    4 when the phase does not change along the row, and enough for an error below
    about 1e-10 otherwise.
    """
    wavenumber = np.atleast_1d(np.asarray(wavenumber, dtype=float))
    opd = ds["opd"].to_numpy()[row : row + 2]
    phase = (
        ds["fringe_phase"].to_numpy()[row : row + 2]
        if "fringe_phase" in ds
        else np.zeros_like(opd)
    )
    littrow = ds["littrow_wavenumber"].to_numpy()[row : row + 2]
    scale, phase_offset = _dispersion(ds, wavenumber)

    def edge_phase(v):
        """Fringe phase at the column edges, at position v along the row."""
        opd_at_v = (1 - v) * opd[0] + v * opd[1]
        phase_at_v = (1 - v) * phase[0] + v * phase[1]
        littrow_at_v = (1 - v) * littrow[0] + v * littrow[1]
        frequency = 2 * np.pi * (wavenumber - littrow_at_v) * scale
        return (
            frequency[np.newaxis, :] * opd_at_v[:, np.newaxis]
            + phase_at_v[:, np.newaxis]
            + phase_offset[np.newaxis, :]
        )

    if num_row_nodes is None:
        num_row_nodes = _row_nodes(np.max(np.abs(edge_phase(1.0) - edge_phase(0.0))))

    nodes, weights = np.polynomial.legendre.leggauss(num_row_nodes)
    result = np.zeros((ds.sizes["pixelcolumn"], len(wavenumber)))
    for v, weight in zip(0.5 * (nodes + 1), 0.5 * weights, strict=True):
        phase_at_v = edge_phase(v)
        left, right = phase_at_v[:-1], phase_at_v[1:]
        result += (
            weight
            * np.cos(0.5 * (left + right))
            * np.sinc((right - left) / (2 * np.pi))
        )

    if "fringe_visibility" in ds:
        result *= ds["fringe_visibility"].to_numpy()[row][:, np.newaxis]
    return result


def _row_nodes(phase_change: float) -> int:
    """Gauss-Legendre nodes for averaging cos over a phase change [rad] along v.

    _MIN_ROW_NODES when the phase is constant along the row; otherwise enough for
    an error below about 1e-10.
    """
    return (
        _MIN_ROW_NODES
        + int(np.ceil(phase_change / 2))
        + int(np.ceil(np.sqrt(phase_change)))
    )


def _dispersion(
    ds: xr.Dataset, wavenumber: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fringe-frequency scale s(nu) and dispersion phase psi(nu)."""
    scale = np.ones_like(wavenumber)
    phase = np.zeros_like(wavenumber)
    if "dispersion_wavenumber" not in ds:
        return scale, phase

    grid = ds["dispersion_wavenumber"].to_numpy()
    order = np.argsort(grid)
    if "fringe_frequency_scale" in ds:
        values = ds["fringe_frequency_scale"].to_numpy()[order]
        scale = np.interp(wavenumber, grid[order], values)
    if "dispersion_phase" in ds:
        values = ds["dispersion_phase"].to_numpy()[order]
        phase = np.interp(wavenumber, grid[order], values)
    return scale, phase


class CalibrationDatabase:
    def __init__(self, ds: xr.Dataset):
        check(ds, "cal_db")
        self._ds = ds

    @classmethod
    def from_file(cls, path: Path) -> CalibrationDatabase:
        return cls(xr.load_dataset(path))

    @property
    def ds(self) -> xr.Dataset:
        return self._ds

    @property
    def num_rows(self) -> int:
        return self._ds.sizes["los"]

    def fingerprint(self) -> str:
        """SHA-256 of the dataset's variables (names, dims, dtypes, values).

        Attributes are left out, so re-saving or annotating the database keeps
        its fingerprint; any change to its values does not.
        """
        digest = hashlib.sha256()
        for name in sorted(self._ds.variables, key=str):
            variable = self._ds.variables[name]
            values = np.ascontiguousarray(variable.to_numpy())
            digest.update(repr((str(name), variable.dims, values.shape)).encode())
            if values.dtype.kind in "OUS":
                digest.update(repr(values.tolist()).encode())
            else:
                digest.update(values.dtype.str.encode())
                digest.update(values.tobytes())
        return digest.hexdigest()

    @property
    def num_columns(self) -> int:
        return self._ds.sizes["pixelcolumn"]

    def sample_wavenumber(self, row: int) -> np.ndarray:
        """L1B sample wavenumbers [cm-1] for one detector row."""
        return self._ds["sample_wavenumber"].isel(los=row).to_numpy()

    def pixel_centre_opd(self, row: int) -> np.ndarray:
        """OPD [cm] at the centre of each pixel in a row (mean of its corners)."""
        edges = self._ds["opd"].to_numpy()[row : row + 2].mean(axis=0)
        return 0.5 * (edges[:-1] + edges[1:])

    def model_wavenumber(self, resolution_cminv: float | None = None) -> np.ndarray:
        """Wavenumber grid [cm-1] covering everywhere the instrument transmits.

        A is zero where spectral_transmission is zero, so radiance on this grid is
        all the forward model needs. The default resolution resolves both the
        fastest fringe (largest |opd|, _FRINGE_SAMPLES_PER_PERIOD samples per
        period) and the transmission table (_TABLE_SAMPLES_PER_STEP samples per
        table step).
        """
        table = self._ds["transmission_wavenumber"].to_numpy()
        if resolution_cminv is None:
            max_opd = np.max(np.abs(self._ds["opd"].to_numpy()))
            resolution_cminv = min(
                1 / (_FRINGE_SAMPLES_PER_PERIOD * max_opd),
                np.min(np.diff(np.sort(table))) / _TABLE_SAMPLES_PER_STEP,
            )
        passband = table[self._ds["spectral_transmission"].to_numpy() > 0]
        # linspace so the grid ends exactly where the transmission does
        num = int(np.ceil((passband.max() - passband.min()) / resolution_cminv)) + 1
        return np.linspace(passband.min(), passband.max(), num)

    def spectral_transmission(self, wavenumber) -> np.ndarray:
        """Spectral transmission at ``wavenumber``; zero outside its table."""
        grid = self._ds["transmission_wavenumber"].to_numpy()
        order = np.argsort(grid)
        values = self._ds["spectral_transmission"].to_numpy()[order]
        return np.interp(wavenumber, grid[order], values, left=0.0, right=0.0)

    def fringe(
        self, row: int, wavenumber, num_row_nodes: int | None = None
    ) -> np.ndarray:
        """V(y,x) · C(y,x,nu): the fringe averaged over each pixel of a row.

        Returns an array of shape (num_columns, len(wavenumber)); see
        pixel_averaged_fringe.
        """
        return pixel_averaged_fringe(self._ds, row, wavenumber, num_row_nodes)

    def radiance_weight(self, wavenumber, bin_width) -> np.ndarray:
        """T(nu) · (dlambda/dnu) · bin_width / E(nu).

        Converts radiance [W m-2 sr-1 nm-1] in spectral bins of width ``bin_width``
        [cm-1] to a photon rate per unit etendue [s-1 m-2 sr-1].
        """
        wavenumber = np.asarray(wavenumber, dtype=float)
        photon_energy_j = PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT_M_S * wavenumber * 100
        return (
            self.spectral_transmission(wavenumber)
            * (1e7 / wavenumber**2)
            * np.asarray(bin_width, dtype=float)
            / photon_energy_j
        )

    def static_dark(self, exposure_time: float) -> np.ndarray:
        """D = dark_offset + t · dark_current [count], shape (num_rows, num_columns)."""
        dark = self._ds["dark_offset"].transpose("los", "pixelcolumn").to_numpy()
        if "dark_current" in self._ds:
            rate = self._ds["dark_current"].transpose("los", "pixelcolumn").to_numpy()
            dark = dark + exposure_time * rate
        return dark

    def forward_operator(
        self, row: int, wavenumber, bin_width, exposure_time: float
    ) -> np.ndarray:
        """A: counts per unit radiance for each pixel of a row and spectral bin.

        Returns an array of shape (num_columns, len(wavenumber)). Excludes the dark
        signal (static_dark), which adds to the image independently of radiance.
        """
        etendue = self._ds["pixel_etendue"].to_numpy()[row]
        gain = float(self._ds["detector_gain"])
        return (
            exposure_time
            / gain
            * etendue[:, np.newaxis]
            * self.radiance_weight(wavenumber, bin_width)[np.newaxis, :]
            * (1 + self.fringe(row, wavenumber))
        )
