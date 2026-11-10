"""SHOW calibration database and the forward model it parameterizes.

The model is defined in the `model` section of src/showlib/schemas/cal_db.yaml.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from showlib.constant import PLANCK_CONSTANT_J_S, SPEED_OF_LIGHT_M_S
from showlib.schemas import check


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

    def spectral_transmission(self, wavenumber) -> np.ndarray:
        """Spectral transmission at ``wavenumber``; zero outside its table."""
        grid = self._ds["transmission_wavenumber"].to_numpy()
        order = np.argsort(grid)
        values = self._ds["spectral_transmission"].to_numpy()[order]
        return np.interp(wavenumber, grid[order], values, left=0.0, right=0.0)

    def fringe(self, row: int, wavenumber, num_row_nodes: int = 4) -> np.ndarray:
        """V(y,x) · C(y,x,nu): the fringe averaged over each pixel of a row.

        Returns an array of shape (num_columns, len(wavenumber)). At any position v
        along the row direction, the fringe phase is linear across the pixel, so
        the average across the pixel is exact: cos(mid-phase) · sinc. The average
        along the row direction uses Gauss-Legendre quadrature with
        ``num_row_nodes`` nodes.
        """
        wavenumber = np.atleast_1d(np.asarray(wavenumber, dtype=float))
        opd = self._ds["opd"].to_numpy()[row : row + 2]
        phase = (
            self._ds["fringe_phase"].to_numpy()[row : row + 2]
            if "fringe_phase" in self._ds
            else np.zeros_like(opd)
        )
        littrow = self._ds["littrow_wavenumber"].to_numpy()[row : row + 2]
        scale, phase_offset = self._dispersion(wavenumber)

        nodes, weights = np.polynomial.legendre.leggauss(num_row_nodes)
        result = np.zeros((self.num_columns, len(wavenumber)))
        for v, weight in zip(0.5 * (nodes + 1), 0.5 * weights, strict=True):
            opd_at_v = (1 - v) * opd[0] + v * opd[1]
            phase_at_v = (1 - v) * phase[0] + v * phase[1]
            littrow_at_v = (1 - v) * littrow[0] + v * littrow[1]

            frequency = 2 * np.pi * (wavenumber - littrow_at_v) * scale
            edge_phase = (
                frequency[np.newaxis, :] * opd_at_v[:, np.newaxis]
                + phase_at_v[:, np.newaxis]
                + phase_offset[np.newaxis, :]
            )
            left, right = edge_phase[:-1], edge_phase[1:]
            result += (
                weight
                * np.cos(0.5 * (left + right))
                * np.sinc((right - left) / (2 * np.pi))
            )

        if "fringe_visibility" in self._ds:
            result *= self._ds["fringe_visibility"].to_numpy()[row][:, np.newaxis]
        return result

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

    def forward_operator(
        self, row: int, wavenumber, bin_width, exposure_time: float
    ) -> np.ndarray:
        """A: counts per unit radiance for each pixel of a row and spectral bin.

        Returns an array of shape (num_columns, len(wavenumber)).
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

    def _dispersion(self, wavenumber: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fringe-frequency scale s(nu) and dispersion phase psi(nu)."""
        scale = np.ones_like(wavenumber)
        phase = np.zeros_like(wavenumber)
        if "dispersion_wavenumber" not in self._ds:
            return scale, phase

        grid = self._ds["dispersion_wavenumber"].to_numpy()
        order = np.argsort(grid)
        if "fringe_frequency_scale" in self._ds:
            values = self._ds["fringe_frequency_scale"].to_numpy()[order]
            scale = np.interp(wavenumber, grid[order], values)
        if "dispersion_phase" in self._ds:
            values = self._ds["dispersion_phase"].to_numpy()[order]
            phase = np.interp(wavenumber, grid[order], values)
        return scale, phase
