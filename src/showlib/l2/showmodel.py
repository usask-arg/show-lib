from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import sasktran2 as sk
import xarray as xr
from skretrieval.core.radianceformat import RadianceGridded
from skretrieval.core.sasktranformat import SASKTRANRadiance
from skretrieval.retrieval.forwardmodel import IdealViewingSpectrograph

from showlib.l2.retrieval import PER_LOS_WF

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
        matrices : np.ndarray or sequence of np.ndarray
            Line-shape matrix of each line of sight, shape (num_samples,
            num_model), mapping radiance on the model grid to L1B radiance at each
            sample; a (num_los, num_samples, num_model) array also works
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
        model_order = _model_order(wavenumber)
        if wavenumber.shape != self.model_wavenumber.shape or not np.allclose(
            wavenumber[model_order], self.model_wavenumber
        ):
            msg = "Radiance is not on the instrument model's model wavenumber grid"
            raise ValueError(msg)

        spectra = radiance.data["radiance"].isel(stokes=0).to_numpy()[model_order]
        modelled_radiance = np.column_stack(
            [matrix @ spectra[:, los] for los, matrix in enumerate(self.matrices)]
        )
        data = xr.Dataset(
            {"radiance": (["wavenumber", "los"], modelled_radiance)},
            coords={"wavenumber": self.sample_wavenumber, "xyz": ["x", "y", "z"]},
        )

        for key in ["look_vectors", "observer_position"]:
            if key in radiance.data:
                name = "los_vectors" if key == "look_vectors" else key
                data[name] = radiance.data[key]

        (spectral,) = radiance.data.wavenumber_cminv.dims

        def model_grid(key):
            """A weighting function as (x, model, los), and its x dimension."""
            wf = radiance.data[key].isel(stokes=0)
            (x_dim,) = (dim for dim in wf.dims if dim not in (spectral, "los"))
            return wf.transpose(x_dim, spectral, "los").to_numpy(), x_dim

        for key in list(radiance.data):
            if key.startswith("wf") and key != PER_LOS_WF:
                wf, x_dim = model_grid(key)
                data[key] = (
                    ["wavenumber", "los", x_dim],
                    self._map_weighting_function(wf, model_order),
                )

        if PER_LOS_WF in radiance.data:
            per_los, _ = model_grid(PER_LOS_WF)
            data["wf"] = (
                ["wavenumber", "los", "x"],
                self._with_per_los_columns(
                    data["wf"].to_numpy(),
                    radiance.data["x"].to_numpy(),
                    self._map_weighting_function(per_los, model_order),
                    radiance.data["x_per_los"].to_numpy(),
                ),
            )

        return RadianceGridded(data)

    @staticmethod
    def _with_per_los_columns(
        dense: np.ndarray,
        dense_index: np.ndarray,
        per_los: np.ndarray,
        per_los_start: np.ndarray,
    ) -> np.ndarray:
        """Full weighting functions (sample, los, state) from SHOWStateVector's
        dense columns and per-line-of-sight elements: element e's value for line
        of sight l is state column per_los_start[e] + l, non-zero only on l."""
        num_samples, num_los, num_per_los = per_los.shape
        wf = np.zeros((num_samples, num_los, len(dense_index) + num_per_los * num_los))
        wf[:, :, dense_index] = dense
        for los in range(num_los):
            wf[:, los, per_los_start + los] = per_los[:, los, :]
        return wf

    def _map_weighting_function(self, wf: np.ndarray, model_order) -> np.ndarray:
        """Each line of sight's matrix applied to ``wf`` (x, model, los), giving
        (sample, los, x).

        One line of sight at a time: wf is the largest array in the retrieval
        (state size by model wavenumbers by lines of sight), so it is not copied.
        """
        result = np.empty((len(self.sample_wavenumber), wf.shape[2], wf.shape[0]))
        for los, matrix in enumerate(self.matrices):
            result[:, los, :] = matrix @ wf[:, model_order, los].T
        return result


def _model_order(wavenumber: np.ndarray) -> slice | np.ndarray:
    """Index putting ``wavenumber`` in increasing order.

    A slice when it is monotonic (SASKTRAN2 returns increasing wavelength), so
    indexing gives a view rather than a copy.
    """
    step = np.diff(wavenumber)
    if np.all(step > 0):
        return slice(None)
    if np.all(step < 0):
        return slice(None, None, -1)
    return np.argsort(wavenumber)


class SHOWForwardModel(IdealViewingSpectrograph):
    def __init__(self, *args, line_shapes: SHOWLineShapes, rows, **kwargs):
        """
        SHOW forward model using B·A line shapes for each line of sight

        Parameters
        ----------
        line_shapes : SHOWLineShapes
            Model grid and line shapes from the calibration database
        rows : sequence
            Detector row of each retrieved line of sight, or the tuple of rows
            averaged into it
        """
        self._line_shapes = line_shapes
        self._rows = list(rows)
        super().__init__(*args, **kwargs)

    def _construct_atmosphere(self):
        """skretrieval's atmosphere, without the unused thermodynamic derivatives.

        skretrieval turns off the pressure and temperature derivatives, but with
        the specific humidity derivative on SASKTRAN2 still returns weighting
        functions for all three. They are not state elements and are dropped
        after the calculation, but at full size (altitude by model wavenumber by
        line of sight) they were about half of SASKTRAN2's output.
        """
        atmo = {}
        for key, geometry in self._model_geometry.items():
            atmo[key] = sk.Atmosphere(
                geometry,
                self._engine_config,
                wavelengths_nm=self._model_wavelength[key],
                pressure_derivative=False,
                temperature_derivative=False,
                specific_humidity_derivative=False,
            )
            self._state_vector.add_to_atmosphere(atmo[key])
            self._ancillary.add_to_atmosphere(atmo[key])
            geometry.refractive_index = (
                sk.optical.refraction.ciddor_index_of_refraction(
                    atmo[key].temperature_k,
                    atmo[key].pressure_pa,
                    np.zeros_like(atmo[key].temperature_k),
                    450,
                    np.nanmean(atmo[key].wavelengths_nm),
                )
            )
        return atmo

    def _construct_model_wavelength(self):
        wavelength = 1e7 / self._line_shapes.model_wavenumber[::-1]
        return dict.fromkeys(self._get_required_wavelength(), wavelength)

    def _construct_inst_model(self):
        return {
            key: self._line_shapes.instrument_model(self._rows, 1e7 / wavelength)
            for key, wavelength in self._get_required_wavelength().items()
        }
