from __future__ import annotations

from showlib import constant


def test_codata_2018_exact_values():
    assert constant.PLANCK_CONSTANT_J_S == 6.62607015e-34
    assert constant.SPEED_OF_LIGHT_M_S == 299792458.0
    assert constant.AVOGADRO_CONSTANT_PER_MOL == 6.02214076e23
    assert constant.MOLAR_GAS_CONSTANT_J_PER_MOL_K == 8.31446261815324
