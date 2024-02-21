from __future__ import annotations

from copy import copy

import numpy as np
import xarray as xr
from skretrieval.retrieval.statevector import StateVectorElement
from skretrieval.retrieval.tikhonov import (
    two_dim_vertical_first_deriv,
)


class BandShifts(StateVectorElement):
    def __init__(self, num_los: int, numerical_delta=0.0001):
        """
        Implements a wavelength shift for every modelled LOS in the forward model

        Parameters
        ----------
        num_los : int
        numerical_delta : float, optional
            _description_, by default 0.0001
        """
        self._shifts = np.zeros(num_los)
        self._numerical_delta = numerical_delta
        super().__init__(True)

    def state(self) -> np.array:
        return copy(self._shifts)

    def name(self) -> str:
        return "band_shifts"

    def lower_bound(self) -> np.array:
        return np.ones_like(self._shifts.flatten()) * -0.1

    def upper_bound(self) -> np.array:
        return np.ones_like(self._shifts.flatten()) * 0.1

    def inverse_apriori_covariance(self) -> np.ndarray:
        gamma = two_dim_vertical_first_deriv(1, len(self._shifts), factor=1e4)
        return gamma.T @ gamma + np.eye(len(self._shifts)) * 1e-5

    def apriori_state(self) -> np.array:
        return np.zeros_like(self.state())

    def propagate_wf(self, radiance) -> np.ndarray:
        wf = np.zeros(
            (
                len(self._shifts),
                radiance["radiance"].shape[0],
                radiance["radiance"].shape[1],
                radiance["radiance"].shape[2],
            )
        )

        for i in range(len(self._shifts)):
            new_rad = radiance.isel(los=i, stokes=0)["radiance"].interp(
                wavelength=radiance.wavelength + self._numerical_delta,
                kwargs={"fill_value": "extrapolate"},
            )

            drad = (
                new_rad - radiance.isel(los=i, stokes=0)["radiance"]
            ) / self._numerical_delta
            wf[i, :, i, 0] = drad
        return xr.DataArray(wf, dims=["x", "wavelength", "los", "stokes"])

    def update_state(self, x: np.array):
        self._shifts = copy(x)

    def modify_input_radiance(self, radiance: xr.Dataset):
        for i in range(len(self._shifts)):
            shift = self._shifts[i]
            for var in list(radiance):
                if "wavelength" in radiance[var].dims:
                    if len(radiance[var].dims) == 3:
                        radiance[var].to_numpy()[:, i, 0] = radiance.isel(
                            los=i, stokes=0
                        )[var].interp(
                            wavelength=radiance.wavelength + shift,
                            kwargs={"fill_value": "extrapolate"},
                        )
                    else:
                        radiance[var].to_numpy()[:, :, i, 0] = radiance.isel(
                            los=i, stokes=0
                        )[var].interp(
                            wavelength=radiance.wavelength + shift,
                            kwargs={"fill_value": "extrapolate"},
                        )

        return radiance


class AltitudeShift(StateVectorElement):
    def __init__(self, numerical_delta=0.0001):
        self._shifts = np.zeros(1)
        self._numerical_delta = numerical_delta
        super().__init__(True)

    def state(self) -> np.array:
        return copy(self._shifts)

    def name(self) -> str:
        return "band_shifts"

    def lower_bound(self) -> np.array:
        return np.ones_like(self._x.flatten()) * -0.3

    def upper_bound(self) -> np.array:
        return np.ones_like(self._x.flatten()) * 0.3

    def inverse_apriori_covariance(self) -> np.ndarray:
        return np.ones_like(self.state()) * 1e-10

    def apriori_state(self) -> np.array:
        return np.zeros_like(self.state())

    def propagate_wf(self, radiance) -> np.ndarray:
        wf = np.zeros(
            (
                len(self._shifts),
                radiance["radiance"].shape[0],
                radiance["radiance"].shape[1],
                radiance["radiance"].shape[2],
            )
        )

        new_rad = (
            radiance.isel(stokes=0)
            .swap_dims({"los": "angle"})["radiance"]
            .interp(
                angle=radiance.angle.to_numpy() + self._numerical_delta,
                kwargs={"fill_value": "extrapolate"},
            )
        )

        drad = (
            new_rad - radiance.isel(stokes=0)["radiance"].to_numpy()
        ) / self._numerical_delta
        wf[0, :, :, 0] = drad
        return xr.DataArray(wf, dims=["x", "wavelength", "los", "stokes"])

    def update_state(self, x: np.array):
        self._shifts = copy(x)

    def modify_input_radiance(self, radiance: xr.Dataset):
        shift = self._shifts[0]

        return (
            radiance.swap_dims({"los": "angle"})
            .interp(
                angle=radiance.angle.to_numpy() + shift,
                kwargs={"fill_value": "extrapolate"},
            )
            .swap_dims({"angle": "los"})
        )
