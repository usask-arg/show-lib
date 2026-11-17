from __future__ import annotations

import numpy as np

from showlib.util import wrap_degrees


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
