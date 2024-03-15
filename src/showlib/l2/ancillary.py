from __future__ import annotations

import numpy as np
import sasktran2 as sk2
import xarray as xr


class SHOWAncillary:
    def __init__(
        self,
        altitudes_m: np.array,
        pressure_pa: np.array,
        temperature_k: np.array,
        aero_data: xr.Dataset | None,
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
        self._aero_data = aero_data

    def add_to_atmosphere(self, atmo: sk2.Atmosphere):
        atmo["rayleigh"] = sk2.constituent.Rayleigh()

        atmo.pressure_pa = np.interp(
            atmo.model_geometry.altitudes(), self._altitudes_m, self._pressure_pa
        )
        atmo.temperature_k = np.interp(
            atmo.model_geometry.altitudes(), self._altitudes_m, self._temperature_k
        )

        if self._aero_data is not None:
            mie_db = sk2.database.MieDatabase(
                sk2.mie.distribution.LogNormalDistribution(),
                sk2.mie.refractive.H2SO4(),
                wavelengths_nm=np.arange(1300, 1400, 10.0),
                median_radius=[70, 80, 90],
                mode_width=[1.5, 1.6, 1.7],
            )

            atmo["aerosol"] = sk2.constituent.NumberDensityScatterer(
                mie_db,
                self._aero_data["altitude"].to_numpy(),
                self._aero_data["number_density"].to_numpy() * 1e6,
                median_radius=self._aero_data["lognormal_median_radius"].to_numpy()
                * 1000,
                mode_width=self._aero_data["lognormal_width"].to_numpy(),
                out_of_bounds_mode="extend",
            )
