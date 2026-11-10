from __future__ import annotations

import numpy as np

from showlib.cal_db import CalibrationDatabase
from showlib.cal_db.ideal import bandpass_filter, ideal_calibration_database


def test_bandpass_filter_shape():
    wavelength = np.array([1360.0, 1362.4, 1364.0, 1365.6, 1368.0])
    transmission = bandpass_filter(
        wavelength,
        centre_nm=1364.0,
        half_width_nm=1.5,
        transition_nm=0.2,
        stopband_db=80,
    )
    np.testing.assert_allclose(transmission[[0, -1]], 1e-4)
    np.testing.assert_allclose(transmission[2], 1.0)
    assert 1e-4 < transmission[1] < 1.0
    assert 1e-4 < transmission[3] < 1.0


def test_builds_a_valid_database():
    cal_db = ideal_calibration_database(num_rows=3, num_columns=64)
    assert isinstance(cal_db, CalibrationDatabase)
    assert cal_db.num_rows == 3
    assert cal_db.num_columns == 64


def test_opd_grid_is_centred_and_uniform():
    cal_db = ideal_calibration_database(num_rows=2, num_columns=64, opd_spacing_cm=0.01)
    centre = cal_db.pixel_centre_opd(0)
    np.testing.assert_allclose(np.diff(centre), 0.01)
    np.testing.assert_allclose(centre.mean(), 0.0, atol=1e-12)


def test_samples_are_on_the_natural_grid_below_littrow_and_in_band():
    num_columns, opd_spacing = 1280, 0.0075
    cal_db = ideal_calibration_database(
        num_rows=2, num_columns=num_columns, opd_spacing_cm=opd_spacing
    )
    littrow = 1e7 / 1362.0
    spacing = 1 / (num_columns * opd_spacing)
    samples = cal_db.sample_wavenumber(1)

    offsets = (littrow - samples) / spacing
    np.testing.assert_allclose(offsets, np.rint(offsets), atol=1e-6)
    assert np.all(offsets >= 1)
    assert np.all(cal_db.spectral_transmission(samples) >= 0.01)
    assert len(samples) > 100


def test_transmission_table_spans_the_filter_with_margin():
    cal_db = ideal_calibration_database(num_rows=1, num_columns=64, table_margin_nm=0.5)
    table_wavelength = 1e7 / cal_db.ds["transmission_wavenumber"].to_numpy()
    assert table_wavelength.min() <= 1364.0 - 1.5 - 0.2 - 0.5 + 1e-6
    assert table_wavelength.max() >= 1364.0 + 1.5 + 0.2 + 0.5 - 1e-6


def test_detector_defaults_follow_the_full_well_and_adc():
    ds = ideal_calibration_database(num_rows=1, num_columns=64).ds
    np.testing.assert_allclose(float(ds["detector_gain"]), 450_000 / 2**12)
    assert float(ds["detector_read_noise"]) == 75.0
