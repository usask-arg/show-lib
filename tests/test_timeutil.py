from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from showlib.timeutil import datetime64_to_datetime, mjd_to_datetime


def test_mjd_to_datetime():
    assert mjd_to_datetime(0) == datetime(1858, 11, 17)
    assert mjd_to_datetime(51544.5) == datetime(2000, 1, 1, 12)
    assert mjd_to_datetime(60276.75) == datetime(2023, 11, 28, 18)


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_datetime64_to_datetime(unit):
    t = np.datetime64("2023-11-28T17:45:12", unit)
    result = datetime64_to_datetime(t)

    assert isinstance(result, datetime)
    assert result == datetime(2023, 11, 28, 17, 45, 12)
