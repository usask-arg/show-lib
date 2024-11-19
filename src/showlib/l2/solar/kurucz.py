from __future__ import annotations

import numpy as np
import pandas as pd
import sasktran2 as sk


class KuruczContinuum:
    def __init__(self):
        db = sk.database.StandardDatabase()

        ir_spectrum_file = db.path("solar/kurucz/irradiance2008/irradabs1560.dat")

        ir_spectrum_data = pd.read_csv(
            ir_spectrum_file.as_posix(), delimiter=r"\s+", header=None, skiprows=12
        )

        self._ir_wavelengths = ir_spectrum_data[0].to_numpy()
        self._ir_spectrum = ir_spectrum_data[1].to_numpy()  # w / m2 / s

        ir_spectrum_file = db.path("solar/kurucz/irradiance2008/irradres1560.dat")

        ir_spectrum_data = pd.read_csv(
            ir_spectrum_file.as_posix(), delimiter=r"\s+", header=None, skiprows=12
        )

        self._ir_res_spectrum = ir_spectrum_data[1].to_numpy()  # w / m2 / s

        energy_to_photons = 6.62607004e-34 * 299792458 / (self._ir_wavelengths * 1e-9)
        self._ir_continuum = (
            self._ir_spectrum / self._ir_res_spectrum / energy_to_photons / 1e4
        )

        vis_spectrum_file = db.path("solar/kurucz/irradiance2005/irradthuwl.dat")

        vis_spectrum_data = pd.read_csv(
            vis_spectrum_file.as_posix(), delimiter=r"\s+", header=None, skiprows=9
        )

        self._vis_wavelengths = vis_spectrum_data[0].to_numpy()
        self._vis_spectrum = vis_spectrum_data[1].to_numpy()  # w / m2 / s

        vis_spectrum_file = db.path("solar/kurucz/irradiance2005/irradrelwl.dat")

        vis_spectrum_data = pd.read_csv(
            vis_spectrum_file.as_posix(), delimiter=r"\s+", header=None, skiprows=5
        )

        self._vis_rel_wavelengths = vis_spectrum_data[0].to_numpy()
        self._vis_rel_spectrum = vis_spectrum_data[1].to_numpy()  # w / m2 / s

        energy_to_photons = 6.62607004e-34 * 299792458 / (self._vis_wavelengths * 1e-9)

        self._vis_continuum = (
            self._vis_spectrum
            / np.interp(
                self._vis_wavelengths, self._vis_rel_wavelengths, self._vis_rel_spectrum
            )
            / energy_to_photons
            / 1e4
        )

    def ir_continuum(self, wavelength_nm: np.array):
        return np.interp(wavelength_nm, self._ir_wavelengths, self._ir_continuum)

    def vis_continuum(self, wavelength_nm: np.array):
        return np.interp(wavelength_nm, self._vis_wavelengths, self._vis_continuum)


if __name__ == "__main__":
    test = KuruczContinuum()
