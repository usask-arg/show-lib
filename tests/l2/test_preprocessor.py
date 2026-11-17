from __future__ import annotations

import numpy as np
import pytest

from showlib.cal_db.ideal import ideal_calibration_database
from showlib.l2.preprocessor import lowest_usable_altitude, weakest_h2o_sample


@pytest.mark.slow
def test_weakest_h2o_sample_is_an_in_band_window():
    cal_db = ideal_calibration_database(2)

    wavenumber = weakest_h2o_sample(cal_db)

    samples = cal_db.sample_wavenumber(0)
    assert np.any(np.isclose(samples, wavenumber, rtol=0, atol=1e-9))
    assert cal_db.spectral_transmission(wavenumber) >= 0.5
    # The window near 7335.2 cm-1 found from simulated L1B
    assert abs(wavenumber - 7335.2) < 0.2


def _cut(ratio, noise=0.0, **kwargs):
    ratio = np.asarray(ratio, dtype=float)
    settings = {
        "smoothing_los": 1,
        "ratio_threshold": 0.1,
        "noise_sigma": 3.0,
        "min_good_rows": 3,
        "fallback_m": 8000.0,
    } | kwargs
    altitude = 1000.0 * np.arange(len(ratio))
    return lowest_usable_altitude(
        altitude, ratio, np.broadcast_to(noise, ratio.shape), **settings
    )


def test_lowest_usable_altitude_is_the_top_of_the_bad_rows_below_good_ones():
    #        0 km  1 km  2 km  3 km ...
    ratio = [0.5, 0.6, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0]

    assert _cut(ratio) == 2000.0


def test_lowest_usable_altitude_ignores_bad_rows_above_the_first_good_run():
    # A cloud at the bottom, then a deviation high up that must not move the cut
    ratio = [0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0]

    assert _cut(ratio) == 1000.0


def test_lowest_usable_altitude_needs_a_run_of_good_rows():
    # A single good row between bad ones does not end the bad region
    ratio = [0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]

    assert _cut(ratio) == 2000.0


def test_lowest_usable_altitude_tolerates_deviations_within_the_noise():
    # 20% deviations with 10% noise are within 3 sigma of the 10% threshold
    ratio = [0.8, 1.2, 0.8, 1.0, 1.0, 1.0]

    assert _cut(ratio, noise=0.0) == 2000.0
    assert _cut(ratio, noise=0.1) == 8000.0


def test_lowest_usable_altitude_still_finds_a_deviation_well_above_the_noise():
    ratio = [0.3, 0.3, 1.0, 1.0, 1.0, 1.0]

    assert _cut(ratio, noise=0.1) == 1000.0


def test_lowest_usable_altitude_smoothing_reduces_the_noise():
    # 25% low over the bottom 10 rows: not significant row by row, but it is
    # once 9 rows are averaged (the noise drops by 3). The averaging blurs the
    # edge: rows 7 to 9 average in enough good rows to pass.
    ratio = [0.75] * 10 + [1.0] * 10

    assert _cut(ratio, noise=0.1) == 8000.0
    assert _cut(ratio, noise=0.1, smoothing_los=9) == 6000.0


def test_lowest_usable_altitude_sorts_by_altitude():
    altitude = np.array([3000.0, 0.0, 2000.0, 1000.0, 4000.0])
    ratio = np.array([1.0, 0.5, 1.0, 0.5, 1.0])

    cut = lowest_usable_altitude(
        altitude,
        ratio,
        np.zeros(5),
        smoothing_los=1,
        ratio_threshold=0.1,
        noise_sigma=3.0,
        min_good_rows=3,
        fallback_m=8000.0,
    )

    assert cut == 1000.0


def test_lowest_usable_altitude_falls_back_when_the_bottom_is_good():
    assert _cut([1.0] * 6) == 8000.0
    # Bad rows only above the first good run are not a low-altitude problem
    assert _cut([1.0, 1.0, 1.0, 0.5, 1.0]) == 8000.0


def test_lowest_usable_altitude_falls_back_without_a_good_run(caplog):
    with caplog.at_level("WARNING"):
        assert _cut([0.5, 1.0, 1.0, 0.5, 0.5]) == 8000.0
    assert "No run of 3 usable rows" in caplog.text


def test_lowest_usable_altitude_treats_missing_ratios_as_bad():
    ratio = [np.nan, np.nan, 1.0, 1.0, 1.0]

    assert _cut(ratio) == 1000.0
