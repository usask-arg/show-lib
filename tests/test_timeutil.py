from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from showlib.timeutil import datetime64_to_datetime, datetime64_to_mjd, mjd_to_datetime


def test_mjd_epoch():
    assert mjd_to_datetime(0) == datetime(1858, 11, 17)
    assert datetime64_to_mjd(np.datetime64("1858-11-17T00:00:00")) == 0.0


def test_j2000():
    assert mjd_to_datetime(51544.5) == datetime(2000, 1, 1, 12)
    assert datetime64_to_mjd(np.datetime64("2000-01-01T12:00:00")) == 51544.5


def test_mjd_round_trip():
    t = np.datetime64("2023-11-28T17:45:12.250000")
    expected = datetime(2023, 11, 28, 17, 45, 12, 250000)
    assert abs(mjd_to_datetime(datetime64_to_mjd(t)) - expected) < timedelta(
        milliseconds=1
    )


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_datetime64_to_datetime(unit):
    t = np.datetime64("2023-11-28T17:45:12", unit)
    result = datetime64_to_datetime(t)

    assert isinstance(result, datetime)
    assert result == datetime(2023, 11, 28, 17, 45, 12)
