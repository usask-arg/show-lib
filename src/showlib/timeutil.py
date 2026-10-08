from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

# Time helpers that used to live in skretrieval.time, which was removed upstream

_MJD_EPOCH = datetime(1858, 11, 17)
_MJD_EPOCH_64 = np.datetime64("1858-11-17T00:00:00.000000")


def mjd_to_datetime(mjd: float) -> datetime:
    """
    Converts a scalar modified julian date to a naive datetime.datetime
    """
    return _MJD_EPOCH + timedelta(days=float(mjd))


def datetime64_to_mjd(utc: np.datetime64) -> float:
    """
    Converts a scalar numpy datetime64 to a floating point modified julian date
    """
    return float(
        (np.datetime64(utc, "us") - _MJD_EPOCH_64) / np.timedelta64(1, "us") / 86400e6
    )


def datetime64_to_datetime(utc: np.datetime64) -> datetime:
    """
    Converts a scalar numpy datetime64 to a naive datetime.datetime
    """
    return np.datetime64(utc, "us").astype(datetime)
