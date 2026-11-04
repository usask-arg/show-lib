"""SHOW instrument line shapes for the L2 retrieval, computed as B·A.

A is the forward model in the calibration database and B is the L1A→L1B inverse,
so the line shape always matches how the L1B was produced.
"""

from __future__ import annotations

import numpy as np

from showlib.cal_db import CalibrationDatabase
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
        row, wavenumber, np.gradient(wavenumber), exposure_time=1.0
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

    @classmethod
    def from_l1b(
        cls,
        cal_db: CalibrationDatabase,
        l1b: L1bDataSet,
        model_res_cminv: float = 0.01,
    ) -> SHOWLineShapes:
        """Line shapes for a granule, using the apodization recorded in its L1B."""
        l1b_wavenumber = l1b.ds["wavenumber"].to_numpy()
        cal_wavenumber = cal_db.ds["sample_wavenumber"].to_numpy()
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
        matrices = []
        for row in rows:
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
            matrices.append(self.matrix(row)[np.argmax(matches, axis=1)])
        return SHOWInstrumentModel(
            np.stack(matrices), requested, self._model_wavenumber
        )
