from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from skretrieval.core.lineshape import LineShape
from skretrieval.core.radianceformat import RadianceGridded
from skretrieval.core.sasktranformat import SASKTRANRadiance
from skretrieval.retrieval.forwardmodel import IdealViewingSpectrograph

if TYPE_CHECKING:
    from showlib.l2.lineshape import SHOWLineShapes


class SHOWBandModel:
    def __init__(self, sample_wvnum: np.array, ils: LineShape):
        """
        A simplified SHOW model that uses a single ILS for all rows, and no FOV integration

        Parameters
        ----------
        sample_wvnum : np.array
            Sample wavenumbers of the instrument in [cm-1]
        ils : LineShape
        """
        self._sample_wvnum = sample_wvnum
        self._ils = ils
        self._interpolator = None
        self._interpolator_cache_wavenumber = None

    def model_radiance(self, radiance: SASKTRANRadiance, ils_scale=None):
        if (
            not np.array_equal(
                radiance.data.wavenumber_cminv.values,
                self._interpolator_cache_wavenumber,
            )
            or ils_scale is not None
        ):
            self._construct_interpolator(
                radiance.data.wavenumber_cminv.values, ils_scale
            )

        modelled_radiance = np.einsum(
            "ij,jk...",
            self._interpolator,
            radiance.data["radiance"].isel(stokes=0).to_numpy(),
            optimize=True,
        )

        data = xr.Dataset(
            {
                "radiance": (["wavenumber", "los"], modelled_radiance),
            },
            coords={
                "wavenumber": self._sample_wvnum,
                "xyz": ["x", "y", "z"],
            },
        )

        if "look_vectors" in radiance.data:
            data["los_vectors"] = radiance.data["look_vectors"]

        if "observer_position" in radiance.data:
            data["observer_position"] = radiance.data["observer_position"]

        for key in list(radiance.data):
            if key.startswith("wf"):
                modelled_wf = np.einsum(
                    "ij,ljk->ikl",
                    self._interpolator,
                    radiance.data[key].isel(stokes=0).to_numpy(),
                    optimize="optimal",
                )

                data[key] = (
                    ["wavenumber", "los", radiance.data[key].dims[0]],
                    modelled_wf,
                )

            # if ils_scale is not None:
            #    modelled_wf[:, -1] = radiance['radiance'].values @ self._d_ils_scale
            # out_ds['wf'] = (['wavelength', 'x'], modelled_wf)

        return RadianceGridded(data)

    def _construct_interpolator(self, wavenumber: np.array, ils_scale=None):
        self._interpolator_cache_wavenumber = copy(wavenumber)

        self._interpolator = np.zeros((len(wavenumber), len(self._sample_wvnum))).T

        if ils_scale is None:
            self._d_ils_scale = None
        else:
            self._d_ils_scale = np.zeros((len(wavenumber), len(self._sample_wvnum)))

        x = wavenumber

        for idx, wvnum in enumerate(self._sample_wvnum):
            self._interpolator[idx, :] = self._ils.integration_weights(wvnum, x)

            if ils_scale is not None:
                d_ils_scale = 0.0001
                self._interpolator[:, idx] = self._ils.integration_weights(
                    wvnum, (x - wvnum) / ils_scale + wvnum
                )
                ilsp = self._ils.integration_weights(
                    wvnum, (x - wvnum) / (ils_scale + d_ils_scale) + wvnum
                )
                self._d_ils_scale[:, idx] = (
                    ilsp - self._interpolator[:, idx]
                ) / d_ils_scale
            else:
                self._interpolator[idx, :] = self._ils.integration_weights(wvnum, x)


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
