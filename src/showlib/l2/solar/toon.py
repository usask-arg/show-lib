from __future__ import annotations

import numpy as np
import pandas as pd

from showlib.config import solar_toon_file


class ToonSolarLines:
    def __init__(self):
        toon_file = solar_toon_file()

        self._data = pd.read_csv(toon_file.as_posix(), header=None, delimiter=r"\s+")
        self._wavelength = 1e7 / self._data[0].to_numpy()[::-1]

        self._transmittance = self._data[1].to_numpy()[::-1]

    def transmittance(self, wavelengths_nm: np.array):
        return np.interp(wavelengths_nm, self._wavelength, self._transmittance)


if __name__ == "__main__":
    test = ToonSolarLines()
