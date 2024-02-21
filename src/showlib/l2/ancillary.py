from __future__ import annotations

import numpy as np
import sasktran2 as sk2


class SHOWAncillary:
    def __init__(
        self, altitudes_m: np.array, pressure_pa: np.array, temperature_k: np.array
    ) -> None:
        """
        Defines the ancillary data necessary for the SHOW retrieval, currently this is only
        Rayleigh scattering, pressure and temperature.

        Parameters
        ----------
        altitudes_m : np.array
            Altitudes the pressure and temperature are defined at
        pressure_pa : np.array
            Pressure at the altitudes in [Pa]
        temperature_k : np.array
            Temperature at the altitudes in [K]
        """
        self._altitudes_m = altitudes_m
        self._pressure_pa = pressure_pa
        self._temperature_k = temperature_k

    def add_to_atmosphere(self, atmo: sk2.Atmosphere):
        atmo["rayleigh"] = sk2.constituent.Rayleigh()

        atmo.pressure_pa = np.interp(
            atmo.model_geometry.altitudes(), self._altitudes_m, self._pressure_pa
        )
        atmo.temperature_k = np.interp(
            atmo.model_geometry.altitudes(), self._altitudes_m, self._temperature_k
        )
