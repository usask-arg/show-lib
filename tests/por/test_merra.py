from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from showlib.por.l2.merra import MERRA2, get_tropopause_height

ALT_KM = np.arange(0, 30.01, 0.5)


def _standard_atmosphere() -> np.ndarray:
    # -6.5 K/km troposphere, isothermal above 11 km, warming above 20 km
    temperature = np.where(ALT_KM <= 11, 288.15 - 6.5 * ALT_KM, 216.65)
    return np.where(ALT_KM > 20, 216.65 + (ALT_KM - 20), temperature)


def test_tropopause_of_standard_atmosphere():
    trop = get_tropopause_height(ALT_KM, _standard_atmosphere())

    # Lapse rate crosses -2 K/km between the 11 km and 11.5 km levels
    assert 11.0 < trop < 11.5


def test_tropopause_follows_temperature_profile():
    shifted = np.interp(ALT_KM, ALT_KM + 3, _standard_atmosphere())
    shifted[ALT_KM < 3] = 288.15 - 6.5 * (ALT_KM[ALT_KM < 3] - 3)

    trop = get_tropopause_height(ALT_KM, shifted)

    assert 14.0 < trop < 14.5


def test_isothermal_profile_has_no_tropopause():
    assert np.isnan(get_tropopause_height(ALT_KM, np.full_like(ALT_KM, 220.0)))


def test_missing_merra_file(tmp_path):
    merra = MERRA2(tmp_path)

    with pytest.raises(OSError, match="20231128"):
        merra.calculate_parameters(
            datetime(2023, 11, 28, 17), 40.0, -100.0, np.arange(0, 30000.0, 1000.0)
        )
