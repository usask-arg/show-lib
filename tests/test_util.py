from __future__ import annotations

import numpy as np
import pytest

from showlib.util import circular_mean_degrees, wrap_degrees


def test_wrap_degrees_into_a_half_open_range():
    angles = np.array([-540.0, -180.0, -90.0, 0.0, 180.0, 270.0, 360.0, 725.0])
    np.testing.assert_allclose(
        wrap_degrees(angles, -180.0), [-180, -180, -90, 0, -180, -90, 0, 5]
    )
    np.testing.assert_allclose(
        wrap_degrees(angles, 0.0), [180, 180, 270, 0, 180, 270, 0, 5]
    )


def test_wrap_degrees_never_reaches_the_upper_limit():
    """An angle just below the lower limit must not round up to lower + 360."""
    assert float(wrap_degrees(-1e-14, 0.0)) == 0.0
    assert float(wrap_degrees(-180.0 - 1e-14, -180.0)) == -180.0


def test_circular_mean_across_the_antimeridian():
    assert circular_mean_degrees([179.0, -177.0]) == pytest.approx(-179.0)
    assert circular_mean_degrees([170.0, -172.0]) == pytest.approx(179.0)


def test_circular_mean_across_the_prime_meridian_in_0_to_360():
    assert circular_mean_degrees([359.0, 1.0, 3.0]) == pytest.approx(1.0)
    assert circular_mean_degrees([350.0, 356.0]) == pytest.approx(-7.0)


def test_circular_mean_matches_the_plain_mean_away_from_the_wrap():
    assert circular_mean_degrees([-100.0, -99.0]) == pytest.approx(-99.5)
