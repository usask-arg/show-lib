from __future__ import annotations

from copy import copy

import numpy as np
import sasktran2 as sk2
import xarray as xr
from skretrieval.core.lineshape import LineShape, UserLineShape
from skretrieval.core.radianceformat import RadianceGridded
from skretrieval.core.sasktranformat import SASKTRANRadiance
from skretrieval.retrieval import ForwardModel

from showlib.l1b.data import L1bImage
from showlib.l2.solar.model import SHOWSolarModel

from .ancillary import SHOWAncillary
from .statevector import SHOWStateVector


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
                    optimize=True,
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


class SHOWFPForwardModel(ForwardModel):
    def __init__(
        self,
        l1b: L1bImage,
        ils: xr.Dataset,
        alt_grid: np.array,
        state_vector: SHOWStateVector,
        ancillary: SHOWAncillary,
        engine_config: sk2.Config,
        model_res: float = 0.01,
    ) -> None:
        super().__init__()

        self._l1 = l1b
        self._state_vector = state_vector
        self._engine_config = engine_config
        self._ancillary = ancillary
        self._ils = ils
        self._alt_grid = alt_grid

        self._model_res = model_res

        self._model_geometry, self._viewing_geo = self._l1.sk2_geometries(
            self._alt_grid
        )

        self._model_wavenumber = self._construct_model_wavenumber()

        self._atmosphere = sk2.Atmosphere(
            self._model_geometry,
            self._engine_config,
            wavenumber_cminv=self._model_wavenumber,
            pressure_derivative=False,
            temperature_derivative=False,
        )

        self._state_vector.add_to_atmosphere(self._atmosphere)
        self._ancillary.add_to_atmosphere(self._atmosphere)

        self._engine = sk2.Engine(
            self._engine_config, self._model_geometry, self._viewing_geo
        )

        self._inst_model = self._construct_inst_model()

        self._solar_model = SHOWSolarModel()

    def _construct_model_wavenumber(self):
        return np.arange(7295, 7340, self._model_res)

    def _construct_inst_model(self):
        delta_wvnum = self._ils.wavenumbers[90000:110000]
        ils = self._ils.ils[90000:110000]

        ls = UserLineShape(delta_wvnum, ils, zero_centered=True)

        wvnum = self._l1.ds.wavenumber.to_numpy()

        return SHOWBandModel(wvnum, ls)

    def calculate_radiance(self):
        sk2_rad = self._engine.calculate_radiance(self._atmosphere)

        solar_irradiance = self._solar_model.irradiance(
            sk2_rad["wavelength"], mjd=54372
        )

        sk2_rad *= xr.DataArray(
            solar_irradiance,
            dims=["wavelength"],
            coords={"wavelength": sk2_rad["wavelength"].to_numpy()},
        )

        sk2_rad["angle"] = (["los"], self._l1.ds.angles.to_numpy())

        sk2_rad = self._state_vector.post_process_sk2_radiances(sk2_rad)

        engine_rad = SASKTRANRadiance.from_sasktran2(sk2_rad)

        l1 = self._inst_model.model_radiance(engine_rad, None)

        l1.data["tangent_altitude"] = (["los"], self._l1.ds.tangent_altitude.to_numpy())
        l1.data["angle"] = (["los"], self._l1.ds.angles.to_numpy())

        return l1
