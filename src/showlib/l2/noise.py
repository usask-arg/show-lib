"""Correlated L1B noise in the L2 retrieval.

Apodization makes the noise of neighbouring L1B samples correlated, but L1B stores
only one-sigma values. The retrieval rebuilds each line of sight's covariance from
those values and the correlation implied by B, and whitens the measurement with
it. skretrieval then sees unit, uncorrelated noise, which works with any of its
minimizers.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import linalg, sparse
from skretrieval.retrieval.measvec import Measurement, MeasurementVector, select

from showlib.l2.lineshape import SHOWLineShapes


def measurement_whitener(
    observation: xr.Dataset, line_shapes: SHOWLineShapes, rows
) -> sparse.csr_matrix:
    """W with W·Sy·Wᵀ = I for the observation's radiances.

    ``observation`` has radiance_noise on (wavenumber, los); the measurement is
    flattened in that order, as skretrieval does. ``rows`` gives each line of
    sight's detector row.
    """
    sigma = observation["radiance_noise"].transpose("wavenumber", "los").to_numpy()
    samples = observation["wavenumber"].to_numpy()
    num_samples, num_los = sigma.shape

    row_index, column_index, values = [], [], []
    for los, row in enumerate(rows):
        correlation = line_shapes.noise_correlation(row, samples)
        covariance = sigma[:, los, None] * correlation * sigma[None, :, los]
        whitener = linalg.solve_triangular(
            np.linalg.cholesky(covariance), np.eye(num_samples), lower=True
        )
        index = np.arange(num_samples) * num_los + los
        block_rows, block_columns = np.meshgrid(index, index, indexing="ij")
        row_index.append(block_rows.ravel())
        column_index.append(block_columns.ravel())
        values.append(whitener.ravel())

    size = num_samples * num_los
    return sparse.csr_matrix(
        (
            np.concatenate(values),
            (np.concatenate(row_index), np.concatenate(column_index)),
        ),
        shape=(size, size),
    )


def whitened_measurement_vector(
    observation: xr.Dataset, line_shapes: SHOWLineShapes, rows
) -> MeasurementVector:
    """All radiances, whitened with the correlated covariance of ``observation``.

    The same whitener is applied to the measured and the modelled radiances and
    to the Jacobian, so the retrieval's chi-square uses the full covariance.
    """
    whitener = measurement_whitener(observation, line_shapes, rows)
    identity = sparse.eye(whitener.shape[0], format="csc")

    def whiten(l1, ctxt, **kwargs):  # noqa: ARG001
        measurement = select(l1, **kwargs)
        if len(measurement.y) != whitener.shape[0]:
            msg = (
                f"Expected a measurement of {whitener.shape[0]} radiances, got "
                f"{len(measurement.y)}"
            )
            raise ValueError(msg)
        if measurement.K is None:
            msg = "Whitening needs a materialized Jacobian"
            raise NotImplementedError(msg)
        return Measurement(
            y=whitener @ measurement.y,
            K=np.asarray(whitener @ measurement.K),
            Sy=identity,
        )

    return MeasurementVector(whiten)
