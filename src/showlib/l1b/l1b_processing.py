from __future__ import annotations

from collections import OrderedDict

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter1d

from .l1b import L1A_DC_Filter as DC_filter
from .l1b import abscal, apodization, get_phase_corrected_spectrum
from .l1b import pixel_response_correction as pixel_response
from .l1b import spectral_response_correction as spectral_response


class level1B_processing:
    """
    Implements Level 1B processing of SHOW interferogram images
    """

    def __init__(
        self,
        specs,
        processing_steps=None,
    ):
        """
        :param specs:
        :param num_samples:
        :param pad_factor:
        """
        if processing_steps is None:
            processing_steps = {
                "apply_phase_correction": False,
                "apply_filter_correction": False,
                "apply_apodization": False,
                "remove_bad_pixels": False,
                "DC_Filter": False,
                "apply_finite_pixel_correction": False,
                "apply_abscal": False,
            }
        self.specs = specs
        self.L1B_options = processing_steps
        self.steps_applied = []
        self._level1_data_processing = OrderedDict()
        self._level1_data_SNR_processing = OrderedDict()
        self.__create_L1_processors__()
        self.__create_L1_SNR_processors__()

    def __create_L1_processors__(self):
        """Creates a L1 processing element"""

        if self.L1B_options["DC_Filter"] is True:
            self.add_component(DC_filter(self.specs), "DC")
            self.steps_applied.append("DC_Filter")

        if self.L1B_options["apply_phase_correction"] is True:
            self.add_component(get_phase_corrected_spectrum(self.specs), "fft")
            self.steps_applied.append("phase_correction")

        if self.L1B_options["apply_filter_correction"] is True:
            self.add_component(spectral_response(self.specs), "spectral_response")
            self.steps_applied.append("filter_spectral_response")

        if self.L1B_options["apply_finite_pixel_correction"] is True:
            self.add_component(pixel_response(self.specs), "pixel_response")
            self.steps_applied.append("finite_pixel_response")

        if self.L1B_options["apply_abscal"] is True:
            self.add_component(abscal(self.specs), "abscal")
            self.steps_applied.append("abscal")

        if self.L1B_options["apply_apodization"] is True:
            self.add_component(apodization(self.specs), "apodization")
            self.steps_applied.append("apodization")

    def __create_L1_SNR_processors__(self):
        """Creates a L1 processing element"""

        if self.L1B_options["DC_Filter"] is True:
            self.add_SNR_component(DC_filter(self.specs), "DC")
            self.steps_applied.append("DC_Filter")

        if self.L1B_options["apply_phase_correction"] is True:
            self.add_SNR_component(get_phase_corrected_spectrum(self.specs), "fft")
            self.steps_applied.append("phase_correction")

        if self.L1B_options["apply_apodization"] is True:
            self.add_SNR_component(apodization(self.specs), "apodization")
            self.steps_applied.append("apodization")

    def add_component(self, processor, name):
        """Adds a component to the processor"""
        self._level1_data_processing[name] = processor

    def add_SNR_component(self, processor, name):
        """Adds a component to the processor"""
        self._level1_data_SNR_processing[name] = processor

    def process_signal(self, l1a_ds):
        signal = l1a_ds.copy(deep=True)
        # add back on the C2 DC bias term
        if self.L1B_options["DC_Filter"] is True:
            signal["image"].data = (l1a_ds["image"].data.T + l1a_ds["C2"].data).T
        else:
            signal["image"].data = l1a_ds["image"].data

        process_array = []
        for _name, processor in self._level1_data_processing.items():
            signal = processor.process_signal(signal)
            process_array.append(signal)

        unfiltered_signal = self.process_SNR_signal(l1a_ds)

        # Calculate the SNR from the phase corrected
        # smooth the standard deviation in the height dimension - there are residual biases in the presence of noise
        STD_NOISE = gaussian_filter1d(
            np.std(np.imag(unfiltered_signal["spectrum"][:, 100:180]), axis=1), 5
        )
        SNR = np.real(unfiltered_signal["spectrum"].T) / STD_NOISE
        radiance_noise = np.real(unfiltered_signal["spectrum"].data.T) / SNR

        return xr.Dataset(
            {
                "radiance": (["wavenumber", "los"], np.real(signal["spectrum"].data.T)),
                "radiance_noise": (["wavenumber", "los"], np.real(radiance_noise.data)),
                "SNR": (["wavenumber", "los"], SNR.data),
            },
            coords={"wavenumber": self.specs.wav_num[0:247]},
        )

    def process_SNR_signal(self, l1a_ds):
        signal = l1a_ds.copy(deep=True)
        # add back on the C2 DC bias term
        if self.L1B_options["DC_Filter"] is True:
            signal["image"].data = (l1a_ds["image"].data.T + l1a_ds["C2"].data).T
        else:
            signal["image"].data = l1a_ds["image"].data
        process_array = []

        for _name, processor in self._level1_data_SNR_processing.items():
            signal = processor.process_signal(signal)
            process_array.append(signal)

        return signal
