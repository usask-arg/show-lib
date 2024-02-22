from __future__ import annotations

from datetime import datetime

import numpy as np

from showlib.l2.data import L2Profile


def test_l2_data_construction():
    alts = np.arange(0, 40000, 250.0)

    h2o_vmr = np.ones(len(alts)) * 1.3

    time = datetime(2021, 1, 1, 0, 0, 0)

    _ = L2Profile(alts, h2o_vmr, 0.0, 0.0, time)
