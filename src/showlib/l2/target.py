from __future__ import annotations

import numpy as np
import xarray as xr
from skretrieval.core.radianceformat import RadianceGridded
from skretrieval.retrieval.statevector import StateVector
from skretrieval.retrieval.target import GenericTarget


class SHOWFPTarget(GenericTarget):
    def __init__(
        self,
        state_vector: StateVector,
        high_alt_norm: bool = False,
        rescale_state_space: bool = False,
        low_alt: float = 11000,
        high_alt: float = 20000,
    ):
        super().__init__(state_vector, rescale_state_space)
        self._high_alt_norm = high_alt_norm
        self._low_alt = low_alt
        self._high_alt = high_alt

    def _internal_measurement_vector(self, l1_data: RadianceGridded):
        result = {}

        if not self._high_alt_norm:
            if "radiance_noise" in l1_data.data:
                result["y"] = l1_data.data["radiance"].to_numpy().flatten()
                result["y_error"] = self._masked_noise(l1_data.data)
            else:
                result["y"] = l1_data.data["radiance"].to_numpy().flatten()

            if "wf" in l1_data.data:
                np_wf = l1_data.data["wf"].to_numpy()
                result["jacobian"] = np_wf.reshape(-1, np_wf.shape[2])

        if self._high_alt_norm:
            md = "los" if "los" in l1_data.data.dims else "angle"

            norm = l1_data.data.where(
                (l1_data.data.tangent_altitude > 20000)
                & (l1_data.data.tangent_altitude < 21000)
            ).mean(dim=md)
            useful = l1_data.data.where(
                (l1_data.data.tangent_altitude < 20000), drop=True
            )

            norm = norm.where(
                (norm.wavenumber > 7312) & (norm.wavenumber < 7318), drop=True
            )
            useful = useful.where(
                (useful.wavenumber > 7312) & (useful.wavenumber < 7318), drop=True
            )

            normalized = useful["radiance"] / norm["radiance"]

            if "radiance_noise" in useful:
                error = (self._masked_noise(useful)) / useful[
                    "radiance"
                ].to_numpy().flatten() ** 2
                result["y_error"] = error.flatten()
                result["y"] = normalized.to_numpy().flatten()

            if "wf" in l1_data.data:
                np_wf_useful = useful["wf"].to_numpy()
                np_wf_norm = norm["wf"].to_numpy()

                np_wf_useful /= norm["radiance"].to_numpy()[:, np.newaxis, np.newaxis]
                np_wf_norm /= norm["radiance"].to_numpy()[:, np.newaxis]
                np_wf_useful -= (np_wf_norm[:, np.newaxis, :]) * normalized.to_numpy()[
                    :, :, np.newaxis
                ]

                result["y"] = normalized.to_numpy().flatten()
                result["jacobian"] = np_wf_useful.reshape(-1, np_wf_useful.shape[2])

        return result

    def _masked_noise(self, l1_data: xr.Dataset):
        f = 1e8

        noise = l1_data["radiance_noise"].to_numpy()

        wavel = l1_data.wavenumber.to_numpy()
        alts = l1_data.tangent_altitude.to_numpy()

        # Test mask
        noise[wavel < 7310, :] *= f
        noise[wavel > 7325, :] *= f

        noise[(wavel > 7314) & (wavel < 7315.5)] *= 100

        if len(alts.shape) == 1:
            noise[:, alts < self._low_alt] *= f
            noise[:, alts > self._high_alt] *= f
        else:
            noise[:, alts[:, 0] < self._low_alt] *= f
            noise[:, alts[:, 0] > self._high_alt] *= f

        return noise.flatten() ** 2
