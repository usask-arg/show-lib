from __future__ import annotations

import numpy as np
import pytest

from showlib.schemas import (
    CONSTRAINT_CHECKS,
    DEFERRED_CONSTRAINTS,
    PRODUCTS,
    load_schema,
    validate,
)


def _constraint_failures(issues):
    return {
        issue.location.removeprefix("constraints.")
        for issue in issues
        if issue.severity == "error" and issue.location.startswith("constraints.")
    }


@pytest.mark.parametrize("product", PRODUCTS)
def test_every_constraint_is_checked_or_explicitly_deferred(product):
    in_schema = {constraint["id"] for constraint in load_schema(product)["constraints"]}
    checked = set(CONSTRAINT_CHECKS.get(product, {}))
    deferred = set(DEFERRED_CONSTRAINTS.get(product, {}))
    assert checked.isdisjoint(deferred)
    assert checked | deferred == in_schema


def test_constraints_skip_when_their_fields_are_missing(l1a):
    issues = validate(l1a.drop_vars(["image", "time"]), "l1a")
    assert _constraint_failures(issues) == set()


# ------------------------------------------------------------------------ L1A
def test_l1a_time_not_increasing(l1a):
    l1a = l1a.isel(time=[1, 0])
    assert _constraint_failures(validate(l1a, "l1a")) == {"time-increasing"}


def test_l1a_image_not_finite(l1a):
    l1a["image"][0, 0, 0] = np.inf
    assert _constraint_failures(validate(l1a, "l1a")) == {"image-finite"}


def test_l1a_detector_matches_calibration(l1a, cal_db):
    assert validate(l1a, "l1a", cal_db=cal_db) == []


def test_l1a_detector_does_not_match_calibration(l1a, cal_db):
    l1a = l1a.isel(pixelcolumn=slice(1, None))
    issues = validate(l1a, "l1a", cal_db=cal_db)
    assert _constraint_failures(issues) == {"detector-matches-calibration"}


# ------------------------------------------------------------------------ L1B
def test_l1b_time_not_increasing(l1b):
    l1b = l1b.isel(time=[1, 0])
    assert _constraint_failures(validate(l1b, "l1b")) == {"time-increasing"}


def test_l1b_wavenumber_not_increasing(l1b):
    l1b = l1b.isel(spectral_sample=[1, 0, 2, 3])
    assert _constraint_failures(validate(l1b, "l1b")) == {"wavenumber-increasing"}


def test_l1b_radiance_not_finite(l1b):
    l1b["radiance_noise"][0, 0, 0] = np.nan
    assert _constraint_failures(validate(l1b, "l1b")) == {"finite-values"}


# --------------------------------------------------------------------- CAL_DB
def test_cal_db_edge_lengths(cal_db):
    cal_db = cal_db.isel(los_edge=slice(1, None))
    assert "edge-lengths" in _constraint_failures(validate(cal_db, "cal_db"))


def test_cal_db_opd_not_monotonic(cal_db):
    cal_db["opd"][1, 3] = cal_db["opd"][1, 5]
    assert _constraint_failures(validate(cal_db, "cal_db")) == {"opd-monotonic"}


def test_cal_db_opd_decreasing_is_monotonic(cal_db):
    cal_db["opd"] = -cal_db["opd"]
    assert validate(cal_db, "cal_db") == []


def test_cal_db_transmission_does_not_span_samples(cal_db):
    cal_db = cal_db.isel(transmission_wavenumber=slice(None, 14))
    failures = _constraint_failures(validate(cal_db, "cal_db"))
    assert failures == {"transmission-covers-samples"}


def test_cal_db_transmission_zero_at_a_sample(cal_db):
    cal_db["spectral_transmission"][:] = 0.0
    failures = _constraint_failures(validate(cal_db, "cal_db"))
    assert "transmission-covers-samples" in failures


def test_cal_db_samples_not_increasing(cal_db):
    cal_db = cal_db.isel(spectral_sample=[1, 0, 2, 3])
    assert "samples-increasing" in _constraint_failures(validate(cal_db, "cal_db"))


def test_cal_db_sample_at_littrow(cal_db):
    cal_db["littrow_wavenumber"][:] = cal_db["sample_wavenumber"][0, 1]
    failures = _constraint_failures(validate(cal_db, "cal_db"))
    assert "samples-off-dc" in failures


def test_cal_db_samples_on_both_sides_of_littrow(cal_db):
    cal_db["littrow_wavenumber"][:] = 7331.25
    failures = _constraint_failures(validate(cal_db, "cal_db"))
    assert failures == {"samples-one-sided"}


def _with_dispersion(cal_db, scale=None, phase=None, include_grid=True):
    wavenumber = np.arange(7320.0, 7346.0, 5.0)
    reference = 7330.0
    offset = wavenumber - reference
    scale = 1 + 1e-3 * offset**2 if scale is None else scale(offset)
    phase = 0.01 * offset**2 if phase is None else phase(offset)
    cal_db = cal_db.assign(
        fringe_frequency_scale=("dispersion_wavenumber", scale, {"units": "1"}),
        dispersion_phase=("dispersion_wavenumber", phase, {"units": "rad"}),
    )
    if include_grid:
        cal_db = cal_db.assign_coords(
            dispersion_wavenumber=(
                "dispersion_wavenumber",
                wavenumber,
                {"units": "cm-1"},
            )
        ).assign(dispersion_reference_wavenumber=((), reference, {"units": "cm-1"}))
    return cal_db


def test_cal_db_valid_dispersion(cal_db):
    assert validate(_with_dispersion(cal_db), "cal_db") == []


def test_cal_db_dispersion_without_grid(cal_db):
    cal_db = _with_dispersion(cal_db, include_grid=False)
    failures = _constraint_failures(validate(cal_db, "cal_db"))
    assert failures == {"dispersion-complete"}


def test_cal_db_dispersion_does_not_cover_band(cal_db):
    cal_db = _with_dispersion(cal_db).isel(dispersion_wavenumber=slice(2, None))
    failures = _constraint_failures(validate(cal_db, "cal_db"))
    assert "dispersion-covers-band" in failures


def test_cal_db_dispersion_scale_not_one_at_reference(cal_db):
    cal_db = _with_dispersion(cal_db, scale=lambda offset: np.full_like(offset, 1.01))
    failures = _constraint_failures(validate(cal_db, "cal_db"))
    assert failures == {"dispersion-reference"}


def test_cal_db_dispersion_phase_has_slope_at_reference(cal_db):
    cal_db = _with_dispersion(cal_db, phase=lambda offset: 0.1 * offset)
    failures = _constraint_failures(validate(cal_db, "cal_db"))
    assert failures == {"dispersion-reference"}
