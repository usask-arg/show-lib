from __future__ import annotations

import numpy as np

from showlib.l2.solar.kurucz import KuruczContinuum
from showlib.l2.solar.toon import ToonSolarLines


class SHOWSolarModel:
    def __init__(self):
        self._continuum = KuruczContinuum()
        self._lines = ToonSolarLines()

        self._ir_cutoff_nm = 1000

    def irradiance(self, wavelengths, solardistance=None, mjd=None):  # noqa: ARG002
        vis_wavel_flag = wavelengths < self._ir_cutoff_nm

        ir_wavel_flag = ~vis_wavel_flag

        irrad = np.zeros_like(wavelengths)

        irrad[vis_wavel_flag] = self._continuum.vis_continuum(
            wavelengths[vis_wavel_flag]
        ) * self._lines.transmittance(wavelengths[vis_wavel_flag])
        irrad[ir_wavel_flag] = self._continuum.ir_continuum(
            wavelengths[ir_wavel_flag]
        ) * self._lines.transmittance(wavelengths[ir_wavel_flag])

        # TODO: Correct for solar distance
        # TODO: Implement doppler shift

        return irrad
