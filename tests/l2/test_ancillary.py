from __future__ import annotations

from sasktran2.util import get_hapi


def test_hapi_is_installed():
    # Ancillary adds a sasktran2 HITRANAbsorber, which needs the optional hapi dependency
    assert get_hapi() is not None
