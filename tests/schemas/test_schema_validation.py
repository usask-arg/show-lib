from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from showlib.schemas import SchemaValidationError, check, validate


def _errors(issues):
    return [issue for issue in issues if issue.severity == "error"]


def _locations(issues):
    return {issue.location for issue in _errors(issues)}


@pytest.mark.parametrize("product", ["l1a", "l1b", "cal_db"])
def test_valid_dataset_has_no_issues(product, request):
    assert validate(request.getfixturevalue(product), product) == []


def test_unknown_product_is_rejected(l1a):
    with pytest.raises(ValueError, match="Unknown product"):
        validate(l1a, "l3")


def test_missing_required_variable(l1a):
    issues = validate(l1a.drop_vars("image"), "l1a")
    assert _locations(issues) == {"variables.image"}


def test_missing_optional_variable_is_allowed(l1a):
    assert validate(l1a.drop_vars("saturated"), "l1a") == []


def test_wrong_dimensions(l1a):
    l1a["image"] = l1a["image"].transpose("time", "pixelcolumn", "los")
    issues = validate(l1a, "l1a")
    assert _locations(issues) == {"variables.image"}
    assert "dims" in _errors(issues)[0].message


def test_wrong_dtype(l1a):
    l1a["exposure_time"] = l1a["exposure_time"].astype(int)
    assert _locations(validate(l1a, "l1a")) == {"variables.exposure_time"}


def test_missing_units(l1a):
    del l1a["exposure_time"].attrs["units"]
    issues = validate(l1a, "l1a")
    assert _locations(issues) == {"variables.exposure_time"}
    assert "units" in _errors(issues)[0].message


def test_wrong_units(l1a):
    l1a["tangent_altitude"].attrs["units"] = "km"
    assert _locations(validate(l1a, "l1a")) == {"variables.tangent_altitude"}


def test_value_below_valid_min(l1a):
    l1a["solar_zenith_angle"][0, 0] = -1.0
    assert _locations(validate(l1a, "l1a")) == {"variables.solar_zenith_angle"}


def test_value_above_valid_max(l1a):
    l1a["relative_solar_azimuth_angle"][0, 0] = 270.0
    assert _locations(validate(l1a, "l1a")) == {
        "variables.relative_solar_azimuth_angle"
    }


def test_nan_is_not_a_range_violation(l1a):
    l1a["solar_zenith_angle"][0, 0] = np.nan
    assert validate(l1a, "l1a") == []


def test_scalar_time_is_rejected(l1a):
    l1a = l1a.isel(time=0)
    assert "variables.time" in _locations(validate(l1a, "l1a"))


def test_unknown_variable_is_a_warning(l1a):
    l1a["mystery"] = l1a["exposure_time"]
    issues = validate(l1a, "l1a")
    assert _errors(issues) == []
    assert [(issue.severity, issue.location) for issue in issues] == [
        ("warning", "variables.mystery")
    ]


def test_missing_required_attribute(l1a):
    del l1a.attrs["schema_version"]
    assert _locations(validate(l1a, "l1a")) == {"attributes.schema_version"}


def test_attribute_not_in_allowed_values(l1b):
    l1b.attrs["apodization_window"] = "hann"
    assert _locations(validate(l1b, "l1b")) == {"attributes.apodization_window"}


def test_required_when_condition_met(l1b):
    del l1b.attrs["apodization_beta"]
    assert _locations(validate(l1b, "l1b")) == {"attributes.apodization_beta"}


def test_required_when_condition_not_met(l1b):
    l1b.attrs["apodization_window"] = "rectangular"
    del l1b.attrs["apodization_beta"]
    assert validate(l1b, "l1b") == []


def test_check_raises_with_every_error(l1a):
    with pytest.raises(SchemaValidationError) as excinfo:
        check(l1a.drop_vars(["image", "exposure_time"]), "l1a")
    message = str(excinfo.value)
    assert "variables.image" in message
    assert "variables.exposure_time" in message


def test_check_passes_with_only_warnings(l1a):
    l1a["mystery"] = l1a["exposure_time"]
    check(l1a, "l1a")


@pytest.mark.parametrize("product", ["l1a", "l1b", "cal_db"])
def test_dataset_still_valid_after_netcdf_round_trip(product, request, tmp_path):
    path = tmp_path / f"{product}.nc"
    request.getfixturevalue(product).to_netcdf(path)
    with xr.open_dataset(path) as ds:
        assert validate(ds, product) == []
