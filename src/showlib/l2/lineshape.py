"""SHOW instrument line shapes for the L2 retrieval, computed as B·A.

A is the forward model in the calibration database and B is the L1A→L1B inverse,
so the line shape always matches how the L1B was produced.
"""

from __future__ import annotations

import numpy as np

from showlib.cal_db import CalibrationDatabase, bin_width
from showlib.l1b.data import L1bDataSet
from showlib.l2.showmodel import SHOWInstrumentModel
from showlib.processing.l1a_to_l1b import inverse_operator

# Requested samples come from the L1B grid, so they match to rounding error
_SAMPLE_MATCH_ATOL_CMINV = 1e-6


def instrument_line_shape(
    cal_db: CalibrationDatabase,
    row: int,
    wavenumber: np.ndarray,
    apodization: float | None,
) -> np.ndarray:
    """B·A for one detector row: shape (num_samples, len(wavenumber)).

    Row k is the L1B radiance of sample k per unit radiance in each bin of
    ``wavenumber``, so ``line_shape @ spectrum`` is what L1A→L1B returns for L1A
    generated from ``spectrum``. It is not renormalized.
    """
    wavenumber = np.asarray(wavenumber, dtype=float)
    forward = cal_db.forward_operator(
        row, wavenumber, bin_width(wavenumber), exposure_time=1.0
    )
    return inverse_operator(cal_db, row, apodization) @ forward


def apodization_from_attrs(attrs: dict) -> float | None:
    """Kaiser beta from L1B processing attributes; None for a rectangular window."""
    window = attrs.get("apodization_window")
    if window == "rectangular":
        return None
    if window == "kaiser":
        return float(attrs["apodization_beta"])
    msg = f"Unknown apodization_window {window!r}"
    raise ValueError(msg)


class SHOWLineShapes:
    """The retrieval's model wavenumber grid and per-row line-shape matrices.

    The grid spans the calibration database's spectral_transmission table at
    ``model_res_cminv``. A is zero wherever the transmission is zero, so nothing
    is truncated, and out-of-band leakage is included. Matrices do not depend on
    the image, so each row is computed once and cached.
    """

    def __init__(
        self,
        cal_db: CalibrationDatabase,
        apodization: float | None,
        model_res_cminv: float = 0.01,
    ):
        self._cal_db = cal_db
        self._apodization = apodization

        table = cal_db.ds["transmission_wavenumber"].to_numpy()
        passband = table[cal_db.ds["spectral_transmission"].to_numpy() > 0]
        self._model_wavenumber = np.arange(
            passband.min(), passband.max() + model_res_cminv / 2, model_res_cminv
        )
        self._matrices: dict[int, np.ndarray] = {}
        self._correlations: dict[int, np.ndarray] = {}

    @classmethod
    def from_l1b(
        cls,
        cal_db: CalibrationDatabase,
        l1b: L1bDataSet,
        model_res_cminv: float = 0.01,
    ) -> SHOWLineShapes:
        """Line shapes for a granule, using the apodization recorded in its L1B."""
        rows = l1b.ds["los"].to_numpy()
        if rows.min() < 0 or rows.max() >= cal_db.num_rows:
            msg = (
                f"The L1B's detector rows ({rows.min()} to {rows.max()}) are not all "
                f"in the calibration database, which has {cal_db.num_rows} rows"
            )
            raise ValueError(msg)
        l1b_wavenumber = l1b.ds["wavenumber"].to_numpy()
        cal_wavenumber = cal_db.ds["sample_wavenumber"].isel(los=rows).to_numpy()
        if l1b_wavenumber.shape != cal_wavenumber.shape or not np.allclose(
            l1b_wavenumber, cal_wavenumber
        ):
            msg = (
                "The L1B wavenumber grid does not match the calibration database's "
                "sample_wavenumber; was the L1B made with a different calibration "
                "database?"
            )
            raise ValueError(msg)
        return cls(cal_db, apodization_from_attrs(l1b.ds.attrs), model_res_cminv)

    @property
    def cal_db(self) -> CalibrationDatabase:
        return self._cal_db

    @property
    def model_wavenumber(self) -> np.ndarray:
        """Model wavenumbers [cm-1], increasing."""
        return self._model_wavenumber

    def matrix(self, row: int) -> np.ndarray:
        """Line-shape matrix of one row: shape (num_samples, num_model)."""
        if row not in self._matrices:
            self._matrices[row] = instrument_line_shape(
                self._cal_db, row, self._model_wavenumber, self._apodization
            )
        return self._matrices[row]

    def instrument_model(self, rows, sample_wavenumber) -> SHOWInstrumentModel:
        """Instrument model for lines of sight on ``rows``, at the given samples.

        The samples are matched to each row's calibration sample wavenumbers and
        returned in increasing order.
        """
        requested = np.sort(np.asarray(sample_wavenumber, dtype=float))
        matrices = [
            self.matrix(row)[self._sample_indices(row, requested)] for row in rows
        ]
        return SHOWInstrumentModel(
            np.stack(matrices), requested, self._model_wavenumber
        )

    def noise_correlation(self, row: int, sample_wavenumber=None) -> np.ndarray:
        """Correlation of L1B radiance noise between the samples of one row.

        L1A→L1B maps pixel noise to radiance through B, so the covariance is
        B·diag(pixel variance)·Bᵀ. Pixel variances are taken as equal across the
        row, which makes the correlation depend only on B. ``sample_wavenumber``
        selects samples (in the given order); the default is all of them.
        """
        if row not in self._correlations:
            operator = inverse_operator(self._cal_db, row, self._apodization)
            covariance = operator @ operator.T
            sigma = np.sqrt(np.diag(covariance))
            self._correlations[row] = covariance / np.outer(sigma, sigma)
        correlation = self._correlations[row]
        if sample_wavenumber is None:
            return correlation
        index = self._sample_indices(row, sample_wavenumber)
        return correlation[np.ix_(index, index)]

    def _sample_indices(self, row: int, sample_wavenumber) -> np.ndarray:
        """Index of each requested wavenumber among the row's calibration samples."""
        requested = np.asarray(sample_wavenumber, dtype=float)
        available = self._cal_db.sample_wavenumber(row)
        matches = np.isclose(
            requested[:, np.newaxis],
            available[np.newaxis, :],
            rtol=0,
            atol=_SAMPLE_MATCH_ATOL_CMINV,
        )
        if not np.all(matches.any(axis=1)):
            missing = requested[~matches.any(axis=1)]
            msg = f"Row {row} has no calibration sample at wavenumbers {missing}"
            raise ValueError(msg)
        return np.argmax(matches, axis=1)
