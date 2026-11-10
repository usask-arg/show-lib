from __future__ import annotations

import numpy as np
import pytest

from showlib.cal_db.ideal import ideal_calibration_database
from showlib.l2.preprocessor import weakest_h2o_sample


@pytest.mark.slow
def test_weakest_h2o_sample_is_an_in_band_window():
    cal_db = ideal_calibration_database(2)

    wavenumber = weakest_h2o_sample(cal_db)

    samples = cal_db.sample_wavenumber(0)
    assert np.any(np.isclose(samples, wavenumber, rtol=0, atol=1e-9))
    assert cal_db.spectral_transmission(wavenumber) >= 0.5
    # The window near 7335.2 cm-1 found from simulated L1B
    assert abs(wavenumber - 7335.2) < 0.2
