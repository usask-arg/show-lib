from __future__ import annotations

from copy import copy

import numpy as np
import xarray as xr
from scipy.interpolate import UnivariateSpline
from skretrieval.retrieval.statevector import StateVectorElement
from skretrieval.retrieval.tikhonov import two_dim_vertical_first_deriv


class MultiplicativeSpline(StateVectorElement):
    def __init__(
        self,
        num_los: int,
        low_wavelength_nm: float,
        high_wavelength_nm: float,
        num_wv: int,
        s: float,
        order=3,
        min_value=0.5,
        max_value=1.5,
    ):
        self._wv = np.linspace(
            low_wavelength_nm, high_wavelength_nm, num_wv, endpoint=True
        )
        self._x = np.ones((num_los, len(self._wv)))
        self._low_wavelength_nm = low_wavelength_nm
        self._high_wavelength_nm = high_wavelength_nm
        self._s = s
        self._order = order
        self._min_value = min_value
        self._max_value = max_value
        super().__init__(True)

    def state(self) -> np.array:
        return self._x.flatten()

    def lower_bound(self) -> np.array:
        return np.ones_like(self._x.flatten()) * self._min_value

    def upper_bound(self) -> np.array:
        return np.ones_like(self._x.flatten()) * self._max_value

    def update_state(self, x: np.array):
        self._x = x.reshape(self._x.shape)

    def inverse_apriori_covariance(self) -> np.ndarray:
        return np.eye(len(self.state())) * 1e-10

    def name(self) -> str:
        return f"spline_{self._low_wavelength_nm}_{self._high_wavelength_nm}"

    def propagate_wf(self, radiance: xr.Dataset) -> xr.Dataset:
        # Calculate the derivative of the spline
        spline_deriv = np.zeros(
            (self._x.shape[0], self._x.shape[1], len(radiance["wavelength"]))
        )

        wv = radiance["wavelength"].to_numpy()
        good = (wv > self._low_wavelength_nm) & (wv < self._high_wavelength_nm)

        for i in range(self._x.shape[0]):
            bx = copy(self._x[i])
            base_spline = UnivariateSpline(self._wv, bx, s=self._s, k=self._order)
            base_vals = base_spline(wv[good])

            for j in range(len(bx)):
                bx[j] += 1e-3
                p_vals = UnivariateSpline(self._wv, bx, s=self._s, k=self._order)(
                    wv[good]
                )
                bx[j] -= 1e-3

                spline_deriv[i, j, good] = (p_vals - base_vals) / 1e-3

        full_deriv = np.zeros(
            (
                spline_deriv.shape[0],
                spline_deriv.shape[1],
                radiance["radiance"].shape[0],
                radiance["radiance"].shape[1],
                radiance["radiance"].shape[2],
            )
        )

        for i in range(self._x.shape[0]):
            full_deriv[i, :, :, i, :] = (
                spline_deriv[i, :, :, np.newaxis]
                * radiance["radiance"].to_numpy()[:, i, :]
            )

        return xr.DataArray(
            full_deriv.reshape(
                (
                    -1,
                    radiance["radiance"].shape[0],
                    radiance["radiance"].shape[1],
                    radiance["radiance"].shape[2],
                )
            ),
            dims=["x", "wavelength", "los", "stokes"],
        )

    def modify_input_radiance(self, radiance: xr.Dataset):
        wv = radiance["wavelength"].to_numpy()
        vals = np.ones((len(wv), self._x.shape[0]))
        good = (wv > self._low_wavelength_nm) & (wv < self._high_wavelength_nm)

        for i in range(self._x.shape[0]):
            base_spline = UnivariateSpline(
                self._wv, self._x[i], s=self._s, k=self._order
            )
            vals[good, i] = base_spline(wv[good])

        radiance *= xr.DataArray(vals, dims=["wavelength", "los"])

        return radiance

    def apriori_state(self) -> np.array:
        return np.ones_like(self.state())


class MultiplicativeSplineOne(StateVectorElement):
    def __init__(
        self,
        low_wavelength_nm: float,
        high_wavelength_nm: float,
        num_wv: int,
        s: float,
        order=3,
        min_value=-100,
        max_value=100,
    ):
        self._wv = np.linspace(
            low_wavelength_nm, high_wavelength_nm, num_wv, endpoint=True
        )
        self._x = np.ones(len(self._wv))
        self._low_wavelength_nm = low_wavelength_nm
        self._high_wavelength_nm = high_wavelength_nm
        self._s = s
        self._order = order
        self._min_value = min_value
        self._max_value = max_value
        super().__init__(True)

    def state(self) -> np.array:
        return self._x.flatten()

    def lower_bound(self) -> np.array:
        return np.ones_like(self._x.flatten()) * self._min_value

    def upper_bound(self) -> np.array:
        return np.ones_like(self._x.flatten()) * self._max_value

    def update_state(self, x: np.array):
        self._x = x.reshape(self._x.shape)

    def inverse_apriori_covariance(self) -> np.ndarray:
        return np.eye(len(self.state())) * 1e-5

    def name(self) -> str:
        return f"spline_{self._low_wavelength_nm}_{self._high_wavelength_nm}"

    def propagate_wf(self, radiance: xr.Dataset) -> xr.Dataset:
        # Calculate the derivative of the spline
        spline_deriv = np.zeros((len(self._x), len(radiance["wavelength"])))

        wv = radiance["wavelength"].to_numpy()
        good = (wv > self._low_wavelength_nm) & (wv < self._high_wavelength_nm)

        bx = copy(self._x)
        base_spline = UnivariateSpline(self._wv, bx, s=self._s, k=self._order)
        base_vals = base_spline(wv[good])

        for j in range(len(bx)):
            bx[j] += 1e-2
            p_vals = UnivariateSpline(self._wv, bx, s=self._s, k=self._order)(wv[good])
            bx[j] -= 1e-2

            spline_deriv[j, good] = (p_vals - base_vals) / 1e-2

        full_deriv = np.zeros(
            (
                spline_deriv.shape[0],
                radiance["radiance"].shape[0],
                radiance["radiance"].shape[1],
                radiance["radiance"].shape[2],
            )
        )

        for i in range(radiance["radiance"].to_numpy().shape[1]):
            full_deriv[:, :, i, :] = (
                spline_deriv[:, :, np.newaxis]
                * radiance["radiance"].to_numpy()[:, i, :]
            )

        return xr.DataArray(
            full_deriv.reshape(
                (
                    -1,
                    radiance["radiance"].shape[0],
                    radiance["radiance"].shape[1],
                    radiance["radiance"].shape[2],
                )
            ),
            dims=["x", "wavelength", "los", "stokes"],
        )

    def modify_input_radiance(self, radiance: xr.Dataset):
        wv = radiance["wavelength"].to_numpy()
        vals = np.ones(len(wv))
        good = (wv > self._low_wavelength_nm) & (wv < self._high_wavelength_nm)

        base_spline = UnivariateSpline(self._wv, self._x, s=self._s, k=self._order)
        vals[good] = base_spline(wv[good])

        for var in list(radiance):
            if "wavelength" in radiance[var].dims:
                radiance[var] *= xr.DataArray(vals, dims=["wavelength"])

        return radiance

    def apriori_state(self) -> np.array:
        return np.ones_like(self.state())


class ScaleFactors(StateVectorElement):
    def __init__(self, num_los: int, min_value=-100, max_value=100):
        self._x = np.ones(num_los)
        self._min_value = min_value
        self._max_value = max_value
        super().__init__(True)

    def state(self) -> np.array:
        return self._x.flatten()

    def lower_bound(self) -> np.array:
        return np.ones_like(self._x.flatten()) * self._min_value

    def upper_bound(self) -> np.array:
        return np.ones_like(self._x.flatten()) * self._max_value

    def update_state(self, x: np.array):
        self._x = x.reshape(self._x.shape)

    def inverse_apriori_covariance(self) -> np.ndarray:
        return np.eye(len(self.state())) * 1e-5

    def name(self) -> str:
        return "scale_factors"

    def propagate_wf(self, radiance: xr.Dataset) -> xr.Dataset:
        full_deriv = np.zeros(
            (
                len(self._x),
                radiance["radiance"].shape[0],
                radiance["radiance"].shape[1],
                radiance["radiance"].shape[2],
            )
        )

        for i in range(radiance["radiance"].to_numpy().shape[1]):
            full_deriv[i, :, i, :] = radiance["radiance"].to_numpy()[:, i, :]

        return xr.DataArray(
            full_deriv.reshape(
                (
                    -1,
                    radiance["radiance"].shape[0],
                    radiance["radiance"].shape[1],
                    radiance["radiance"].shape[2],
                )
            ),
            dims=["x", "wavelength", "los", "stokes"],
        )

    def modify_input_radiance(self, radiance: xr.Dataset):
        vals = self._x

        for var in list(radiance):
            if "wavelength" in radiance[var].dims:
                radiance[var] *= xr.DataArray(vals, dims=["los"])

        return radiance

    def apriori_state(self) -> np.array:
        return np.ones_like(self.state())


class ScaleFactorsPoly(StateVectorElement):
    def __init__(self, num_los: int, min_value=-100, max_value=100, order=1):
        self._x = np.ones((num_los, order + 1))
        self._x[:, 1:] = 0
        self._min_value = min_value
        self._max_value = max_value
        self._order = order
        super().__init__(True)

    def state(self) -> np.array:
        return self._x.flatten()

    def lower_bound(self) -> np.array:
        return np.ones_like(self._x.flatten()) * self._min_value

    def upper_bound(self) -> np.array:
        return np.ones_like(self._x.flatten()) * self._max_value

    def update_state(self, x: np.array):
        self._x = x.reshape(self._x.shape)

    def inverse_apriori_covariance(self) -> np.ndarray:
        return np.eye(len(self.state())) * 1e-5

    def name(self) -> str:
        return "scale_factors"

    def propagate_wf(self, radiance: xr.Dataset) -> xr.Dataset:
        w = radiance["wavelength"].to_numpy() - radiance["wavelength"].to_numpy()[0]
        full_deriv = np.zeros(
            (
                self._x.shape[0],
                self._x.shape[1],
                radiance["radiance"].shape[0],
                radiance["radiance"].shape[1],
                radiance["radiance"].shape[2],
            )
        )

        for i in range(radiance["radiance"].to_numpy().shape[1]):
            for o in range(self._order + 1):
                full_deriv[i, o, :, i, :] = (
                    radiance["radiance"].to_numpy()[:, i, :] * (w**o)[:, np.newaxis]
                )

        return xr.DataArray(
            full_deriv.reshape(
                (
                    -1,
                    radiance["radiance"].shape[0],
                    radiance["radiance"].shape[1],
                    radiance["radiance"].shape[2],
                )
            ),
            dims=["x", "wavelength", "los", "stokes"],
        )

    def modify_input_radiance(self, radiance: xr.Dataset):
        vals = self._x
        w = radiance["wavelength"].to_numpy() - radiance["wavelength"].to_numpy()[0]

        mvals = np.zeros((len(w), self._x.shape[0]))

        mvals[:] = vals[:, 0][np.newaxis, :]

        for o in range(1, self._order + 1):
            mvals += vals[:, o][np.newaxis, :] * (w**o)[:, np.newaxis]

        for var in list(radiance):
            if "wavelength" in radiance[var].dims:
                radiance[var] *= xr.DataArray(mvals, dims=["wavelength", "los"])

        return radiance

    def apriori_state(self) -> np.array:
        x_a = np.zeros_like(self._x)
        x_a[:, 0] = 1
        return x_a.flatten()


class AddFactors(StateVectorElement):
    def __init__(self, num_los: int, min_value=-10, max_value=10, order=0):
        self._scale = 1e11
        self._x = np.zeros((num_los, order + 1))
        self._min_value = min_value
        self._max_value = max_value
        self._order = order
        super().__init__(True)

    def state(self) -> np.array:
        return self._x.flatten()

    def lower_bound(self) -> np.array:
        return np.ones_like(self._x.flatten()) * self._min_value

    def upper_bound(self) -> np.array:
        return np.ones_like(self._x.flatten()) * self._max_value

    def update_state(self, x: np.array):
        self._x = x.reshape(self._x.shape)

    def inverse_apriori_covariance(self) -> np.ndarray:
        gamma = two_dim_vertical_first_deriv(1, len(self.state()), factor=100)
        return gamma.T @ gamma + np.eye(len(self.state())) * 1

    def name(self) -> str:
        return "add_factors"

    def propagate_wf(self, radiance: xr.Dataset) -> xr.Dataset:
        full_deriv = np.zeros(
            (
                self._x.shape[0],
                self._x.shape[1],
                radiance["radiance"].shape[0],
                radiance["radiance"].shape[1],
                radiance["radiance"].shape[2],
            )
        )
        w = radiance["wavelength"].to_numpy() - radiance["wavelength"].to_numpy()[0]

        for i in range(radiance["radiance"].to_numpy().shape[1]):
            for o in range(self._order + 1):
                full_deriv[i, o, :, i, 0] = self._scale * w**o

        return xr.DataArray(
            full_deriv.reshape(
                (
                    -1,
                    radiance["radiance"].shape[0],
                    radiance["radiance"].shape[1],
                    radiance["radiance"].shape[2],
                )
            ),
            dims=["x", "wavelength", "los", "stokes"],
        )

    def modify_input_radiance(self, radiance: xr.Dataset):
        vals = self._x
        w = radiance["wavelength"].to_numpy() - radiance["wavelength"].to_numpy()[0]

        mvals = np.zeros((len(w), self._x.shape[0]))

        mvals[:] = vals[:, 0][np.newaxis, :]

        for o in range(1, self._order + 1):
            mvals += vals[:, o][np.newaxis, :] * (w**o)[:, np.newaxis] * self._scale

        radiance["radiance"] += xr.DataArray(mvals, dims=["wavelength", "los"])

        return radiance

    def apriori_state(self) -> np.array:
        return np.zeros_like(self.state())
