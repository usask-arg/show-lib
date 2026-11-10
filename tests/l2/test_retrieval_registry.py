from __future__ import annotations

import pytest
from skretrieval.retrieval.processing import Retrieval

import showlib.processing.l1b_to_l2  # noqa: F401
from showlib.l2.retrieval import (
    SHOWRetrieval,
    aerosol_extinction_profile,
    h2o_ret_optical_property,
    stratospheric_aerosol_optical_property,
)

PER_SUBCLASS_REGISTRIES = (
    SHOWRetrieval._optical_property_fns is not Retrieval._optical_property_fns
)


def test_optical_properties_registered_on_show_retrieval():
    fns = SHOWRetrieval._optical_property_fns
    assert fns["h2o"] is h2o_ret_optical_property
    assert fns["stratospheric_aerosol"] is stratospheric_aerosol_optical_property


def test_aerosol_state_registered_on_show_retrieval():
    state = SHOWRetrieval._state_fns["aerosols"]["extinction_profile_2"]
    assert state is aerosol_extinction_profile


@pytest.mark.skipif(
    not PER_SUBCLASS_REGISTRIES,
    reason="the installed skretrieval shares one registry across Retrieval subclasses",
)
def test_registrations_do_not_replace_shared_ones():
    # Other packages (e.g. aliprocessing) register these names too; importing
    # showlib must not change what their retrievals resolve
    assert Retrieval._optical_property_fns.get("h2o") is not h2o_ret_optical_property
    assert (
        Retrieval._optical_property_fns.get("stratospheric_aerosol")
        is not stratospheric_aerosol_optical_property
    )
    assert "extinction_profile_2" not in Retrieval._state_fns["aerosols"]
