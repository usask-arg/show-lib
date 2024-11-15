from __future__ import annotations

from copy import copy

import numpy as np
import xarray as xr
from skretrieval.core.lineshape import LineShape
from skretrieval.core.radianceformat import RadianceGridded
from skretrieval.core.sasktranformat import SASKTRANRadiance
from skretrieval.retrieval.forwardmodel import IdealViewingSpectrograph

from showlib.l2.solar.model import SHOWSolarModel


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


class SHOWForwardModel(IdealViewingSpectrograph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._solar_model = SHOWSolarModel()

    def calculate_radiance(self):
        l1 = {}
        for key in self._engine:
            sk2_rad = self._engine[key].calculate_radiance(self._atmosphere[key])

            solar_irradiance = self._solar_model.irradiance(
                sk2_rad["wavelength"], mjd=54372
            )

            sk2_rad *= xr.DataArray(
                solar_irradiance,
                dims=["wavelength"],
                coords={"wavelength": sk2_rad["wavelength"].to_numpy()},
            )

            sk2_rad = self._state_vector.post_process_sk2_radiances(sk2_rad)
            sk2_rad = SASKTRANRadiance.from_sasktran2(sk2_rad)

            l1[key] = self._inst_model[key].model_radiance(sk2_rad, None)
            l1[key].data = l1[key].data.reindex(
                wavenumber=l1[key].data.wavenumber[::-1]
            )

            self._observation.append_information_to_l1(l1)

        return l1
