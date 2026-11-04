from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from skretrieval.core.radianceformat import RadianceGridded
from skretrieval.core.sasktranformat import SASKTRANRadiance
from skretrieval.retrieval.forwardmodel import IdealViewingSpectrograph

if TYPE_CHECKING:
    from showlib.l2.lineshape import SHOWLineShapes


class SHOWInstrumentModel:
    def __init__(
        self,
        matrices: np.ndarray,
        sample_wavenumber: np.ndarray,
        model_wavenumber: np.ndarray,
    ):
        """
        SHOW instrument model with a separate line-shape matrix for each line of sight

        Parameters
        ----------
        matrices : np.ndarray
            Line-shape matrices, shape (num_los, num_samples, num_model), mapping
            radiance on the model grid to L1B radiance at each sample
        sample_wavenumber : np.ndarray
            Sample wavenumbers in [cm-1], increasing
        model_wavenumber : np.ndarray
            Model wavenumbers in [cm-1], increasing
        """
        self.matrices = matrices
        self.sample_wavenumber = sample_wavenumber
        self.model_wavenumber = model_wavenumber

    def model_radiance(self, radiance: SASKTRANRadiance, _=None) -> RadianceGridded:
        wavenumber = radiance.data.wavenumber_cminv.to_numpy()
        order = np.argsort(wavenumber)
        if wavenumber.shape != self.model_wavenumber.shape or not np.allclose(
            wavenumber[order], self.model_wavenumber
        ):
            msg = "Radiance is not on the instrument model's model wavenumber grid"
            raise ValueError(msg)

        modelled_radiance = np.einsum(
            "lsm,ml->sl",
            self.matrices,
            radiance.data["radiance"].isel(stokes=0).to_numpy()[order],
            optimize=True,
        )
        data = xr.Dataset(
            {"radiance": (["wavenumber", "los"], modelled_radiance)},
            coords={"wavenumber": self.sample_wavenumber, "xyz": ["x", "y", "z"]},
        )

        for key in ["look_vectors", "observer_position"]:
            if key in radiance.data:
                name = "los_vectors" if key == "look_vectors" else key
                data[name] = radiance.data[key]

        for key in list(radiance.data):
            if key.startswith("wf"):
                wf = radiance.data[key].isel(stokes=0)
                data[key] = (
                    ["wavenumber", "los", wf.dims[0]],
                    np.einsum(
                        "lsm,xml->slx",
                        self.matrices,
                        wf.to_numpy()[:, order],
                        optimize=True,
                    ),
                )

        return RadianceGridded(data)


class SHOWForwardModel(IdealViewingSpectrograph):
    def __init__(self, *args, line_shapes: SHOWLineShapes, rows: np.ndarray, **kwargs):
        """
        SHOW forward model using B·A line shapes for each line of sight

        Parameters
        ----------
        line_shapes : SHOWLineShapes
            Model grid and line shapes from the calibration database
        rows : np.ndarray
            Detector row of each retrieved line of sight
        """
        self._line_shapes = line_shapes
        self._rows = np.asarray(rows)
        super().__init__(*args, **kwargs)

    def _construct_model_wavelength(self):
        wavelength = 1e7 / self._line_shapes.model_wavenumber[::-1]
        return dict.fromkeys(self._get_required_wavelength(), wavelength)

    def _construct_inst_model(self):
        return {
            key: self._line_shapes.instrument_model(self._rows, 1e7 / wavelength)
            for key, wavelength in self._get_required_wavelength().items()
        }
