from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from showlib.l1b.l1b import (
    DC_Filter,
    abscal,
    apodization,
    bad_pixel_removal,
    get_phase_corrected_spectrum,
    pixel_response_correction,
    shs_spectrum,
    spectral_response_correction,
)
from showlib.l1b.l1b_processing import level1B_processing

ALL_STEPS = {
    "apply_phase_correction": True,
    "apply_filter_correction": True,
    "apply_apodization": True,
    "remove_bad_pixels": True,
    "DC_Filter": True,
    "apply_finite_pixel_correction": True,
    "apply_abscal": True,
}


def test_interp_bad_pixels():
    image = np.tile(np.arange(5.0)[:, np.newaxis], (1, 3))
    image[2, 1] = np.nan
    image[0, 2] = np.nan

    result = bad_pixel_removal().interp_bad_pixels(image.copy())

    np.testing.assert_array_equal(result[:, 1], np.arange(5.0))
    # Leading NaNs are filled with the nearest good value
    np.testing.assert_array_equal(result[:, 2], [1.0, 1.0, 2.0, 3.0, 4.0])
    # Column 0 is replaced by the (corrected) column 1
    np.testing.assert_array_equal(result[:, 0], result[:, 1])


def test_bad_pixel_map_is_flipped():
    image = np.tile(np.arange(5.0)[:, np.newaxis], (1, 3))
    bad_pixel_map = np.ones_like(image)
    bad_pixel_map[0, 2] = np.nan  # flipud puts this at row 4

    signal = xr.Dataset(
        {
            "image": (["pixelheight", "pixelcolumn"], image * 10),
            "bad_pixel_map": (["pixelheight", "pixelcolumn"], bad_pixel_map),
        }
    )
    result = bad_pixel_removal().process_signal(signal)

    # The flagged pixel at the edge is filled from its nearest neighbour
    np.testing.assert_array_equal(result["image"][:, 2], [0, 10, 20, 30, 30])
    assert not np.isnan(result["image"]).any()


def test_dc_filter_function():
    dc = DC_Filter()
    s = 230
    f = np.linspace(0, s, 50)
    response = dc.dc_filt_func(f, s, N=4)

    assert response[0] == pytest.approx(1.0)
    assert response[-1] == pytest.approx(0.0)
    assert np.all(np.diff(response) <= 0)


def test_box_car():
    box = get_phase_corrected_spectrum().box_car(10, 2, 5)

    np.testing.assert_array_equal(box, [0, 0, 0, 1, 1, 1, 1, 1, 0, 0])


@pytest.mark.parametrize("n", [64, 65])
def test_wavenumber_scale(n):
    spacing = 0.01
    scale = shs_spectrum(specs=None).get_wavenumber_scale(n, spacing)

    assert len(scale) == n
    assert scale[n // 2] == 0
    np.testing.assert_allclose(np.diff(scale), 1 / (n * spacing))


def test_full_spectrum_peak():
    n = 128
    spacing = 0.01
    f0 = 10 / (n * spacing)
    x = (np.arange(n) - n // 2) * spacing
    interferogram = np.cos(2 * np.pi * f0 * x)

    spectrum, freqs = get_phase_corrected_spectrum().get_full_spectrum(
        spacing, interferogram, pad_factor=0
    )

    assert len(spectrum) == len(freqs) == n
    peaks = freqs[np.argsort(np.abs(spectrum))[-2:]]
    np.testing.assert_allclose(sorted(peaks), [-f0, f0])


@pytest.mark.parametrize("max_opd", [0.05, 0.0731])
def test_hanning_apodization_kernel(max_opd):
    apo = apodization()
    apo.specs = {"max_opd": max_opd, "wav_num": np.arange(7295, 7340, 0.5)}

    wavenumbers, kernel = apo.apodization_function()

    assert np.all(np.isfinite(kernel))
    centre = np.argmin(np.abs(wavenumbers))
    assert wavenumbers[centre] == pytest.approx(0, abs=1e-9)
    assert kernel[centre] == pytest.approx(0.5)
    np.testing.assert_allclose(kernel, kernel[::-1], atol=1e-12)


def test_hanning_apodization_kernel_singularity():
    # With max_opd = 0.05 the kernel grid lands exactly on 2 * L * x = +-1
    apo = apodization()
    apo.specs = {"max_opd": 0.05, "wav_num": np.arange(7295, 7340, 0.5)}

    wavenumbers, kernel = apo.apodization_function()

    singular = np.isclose(np.abs(2 * 0.05 * wavenumbers), 1)
    assert singular.sum() == 2
    np.testing.assert_allclose(kernel[singular], 0.25)
    assert apo.apodization_function(type="boxcar") is None


@pytest.mark.parametrize(
    ("processor", "variable", "expected"),
    [
        (spectral_response_correction, "filter_shape", 2.0),
        (pixel_response_correction, "pixel_response", 2.0),
        (abscal, "abs_cal", 8.0),
    ],
)
def test_spectral_corrections(processor, variable, expected):
    signal = xr.Dataset(
        {
            "spectrum": (["los", "wavenumber"], np.full((2, 3), 4.0)),
            variable: (["wavenumber"], np.full(3, 2.0)),
        }
    )

    result = processor().process_signal(signal)

    np.testing.assert_array_equal(result["spectrum"], expected)


def test_level1b_processing_defaults_to_no_steps():
    proc = level1B_processing()

    assert proc.steps_applied == []
    assert proc.SNR_steps_applied == []
    assert len(proc._level1_data_processing) == 0


def test_level1b_processing_step_order():
    proc = level1B_processing(processing_steps=ALL_STEPS)

    assert proc.steps_applied == [
        "Bad_Pixel",
        "DC_Filter",
        "phase_correction",
        "filter_spectral_response",
        "finite_pixel_response",
        "abscal",
        "apodization",
    ]
    # The SNR chain skips the radiometric corrections
    assert proc.SNR_steps_applied == [
        "Bad_Pixel",
        "DC_Filter",
        "phase_correction",
        "apodization",
    ]
