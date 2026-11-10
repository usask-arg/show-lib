from __future__ import annotations

import numpy as np
import xarray as xr

from showlib.schemas import apply_metadata, load_schema, validate


def _strip_metadata(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()
    for var in ds.variables.values():
        var.attrs.clear()
    ds.attrs.pop("schema_version", None)
    return ds


def test_fills_in_units_and_schema_version(l1a):
    result = apply_metadata(_strip_metadata(l1a), "l1a")
    assert validate(result, "l1a") == []
    assert result["image"].attrs["units"] == "count"
    assert result.attrs["schema_version"] == load_schema("l1a")["schema_version"]


def test_adds_descriptions(l1b):
    result = apply_metadata(l1b, "l1b")
    expected = load_schema("l1b")["variables"]["radiance"]["description"]
    assert result["radiance"].attrs["description"] == expected


def test_does_not_modify_its_input(l1a):
    stripped = _strip_metadata(l1a)
    apply_metadata(stripped, "l1a")
    assert "units" not in stripped["image"].attrs
    assert "schema_version" not in stripped.attrs


def test_keeps_existing_units_so_validation_still_catches_them(l1a):
    l1a["tangent_altitude"].attrs["units"] = "km"
    result = apply_metadata(l1a, "l1a")
    assert result["tangent_altitude"].attrs["units"] == "km"
    assert {issue.location for issue in validate(result, "l1a")} == {
        "variables.tangent_altitude"
    }


def test_leaves_unknown_variables_alone(l1a):
    l1a["mystery"] = l1a["exposure_time"].copy()
    l1a["mystery"].attrs.clear()
    assert apply_metadata(l1a, "l1a")["mystery"].attrs == {}


def test_fills_in_group_metadata(l1a, l1a_simulation):
    tree = xr.DataTree.from_dict(
        {"/": _strip_metadata(l1a), "/simulation": _strip_metadata(l1a_simulation)}
    )
    result = apply_metadata(tree, "l1a")
    assert validate(result, "l1a") == []
    assert result["simulation"]["image_noiseless"].attrs["units"] == "count"


def test_keeps_nested_groups(l1a, l1a_simulation):
    nested = xr.Dataset({"extra": ("x", [1.0, 2.0])})
    tree = xr.DataTree.from_dict(
        {"/": l1a, "/simulation": l1a_simulation, "/simulation/inner": nested}
    )
    result = apply_metadata(tree, "l1a")
    np.testing.assert_array_equal(result["simulation/inner"]["extra"], [1.0, 2.0])
