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
    def __init__(
        self,
        *args,
        line_shapes: SHOWLineShapes,
        rows,
        line_shape_tolerance: float | None = None,
        **kwargs,
    ):
        """
        SHOW forward model using B·A line shapes for each line of sight

        Parameters
        ----------
        line_shapes : SHOWLineShapes
            Model grid and line shapes from the calibration database
        rows : sequence
            Detector row of each retrieved line of sight, or the tuple of rows
            averaged into it
        line_shape_tolerance : float, optional
            If given, the radiative transfer runs only over the window of the model
            grid that the measured samples' line shapes need, to within this
            fraction of their weight (see SHOWLineShapes.instrument_model).
            Otherwise it runs over the whole model grid.
        """
        self._line_shapes = line_shapes
        self._rows = list(rows)
        self._line_shape_tolerance = line_shape_tolerance
        self._inst_models = None
        super().__init__(*args, **kwargs)

    def _instrument_models(self) -> dict[str, SHOWInstrumentModel]:
        """The instrument models, which also set the model grid; built once."""
        if self._inst_models is None:
            self._inst_models = {
                key: self._line_shapes.instrument_model(
                    self._rows, 1e7 / wavelength, self._line_shape_tolerance
                )
                for key, wavelength in self._get_required_wavelength().items()
            }
        return self._inst_models

    def _construct_model_wavelength(self):
        return {
            key: 1e7 / model.model_wavenumber[::-1]
            for key, model in self._instrument_models().items()
        }

    def _construct_inst_model(self):
        return self._instrument_models()
