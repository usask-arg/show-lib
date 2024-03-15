from __future__ import annotations

from .data import L1aImage


def add_dcbias(l1a: L1aImage):
    l1a.ds["image"] += l1a.ds["C2"]

    return l1a
