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
