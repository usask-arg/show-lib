from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.optimize import least_squares
from skretrieval.core.lineshape import UserLineShape


class bad_pixel_removal:
    """
    Used to correct bad pixels"""

    def __init__(self, mode="linear"):
        self.mode = mode

    def interp_bad_pixels(self, image):
        """
        Interpolate bad pixels
        :param image: input image
        :return: output image with bad pixels interpolated

        """

        for i in range(np.shape(image)[1]):
            # get indices of nans
            index = np.where(np.isnan(image[:, i]))
            # Do a simple linear interpolation for now if there are nans
            if len(index[0]) > 0:
                good_index = np.where(~np.isnan(image[:, i]))
                image[index, i] = np.interp(
                    index[0], good_index[0], image[good_index[0], i]
                )
            # make everything outside the SHOW window Nans
            # row 0 is off in the science flight.
            image[:, 0] = image[:, 1]
        return image

    def process_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Process the signal
        :param signal: input signal
        :return: processed output signal
        """

        signal["image"].data = self.interp_bad_pixels(
            signal["image"].data * signal["bad_pixel_map"].data
        )

        return signal


class apodization:
    def __init__(self):
        self.specs = []

    def apodization_function(self, type="hanning"):
        if type == "hanning":
            L = self.specs["max_opd"]
            sample_spacing = np.mean(np.diff(self.specs["wav_num"]))
            centered_wavenumbers = np.arange(
                -10, 10 + sample_spacing / 10, sample_spacing / 10
            )
            apo = (
                0.5
                * np.sinc(2 * L * centered_wavenumbers)
                / (1 - (2 * L * centered_wavenumbers) ** 2)
            )
            return centered_wavenumbers, apo
        return None

    def _construct_wavnum_interpolator(self, center_wavenumber, hires_wavenumber):
        wavenumber_interpolator = np.zeros((1, len(hires_wavenumber)))

        wavenumber_interpolator[0, :] = self.apodization_lineshape.integration_weights(
            center_wavenumber, hires_wavenumber
        )

        return wavenumber_interpolator

    def _construct_wavnum_interpolation(self, center_wavenumber, hires_wavenumber):
        return self._construct_wavnum_interpolator(center_wavenumber, hires_wavenumber)

    def process_signal(self, signal):
        data_image = signal["spectrum"].data
        specs = {}
        specs["max_opd"] = np.max(signal["opd_x"].data)
        specs["wav_num"] = signal["wavenumbers"].data
        self.specs = specs
        # convolve he apodization function with the input signal
        apo = self.apodization_function()
        self.apodization_lineshape = UserLineShape(apo[0], apo[1], zero_centered=True)

        # set the shs spectral sampling
        # Set up the wavelength interpolator
        wavel_interp = []
        for s in self.specs["wav_num"]:
            interp_vals = self._construct_wavnum_interpolation(s, self.specs["wav_num"])
            wavel_interp.append(interp_vals)

        wavel_interp = np.array(wavel_interp)
        signal["spectrum"].data = np.einsum(
            "ij,jk...", data_image, wavel_interp[:, 0, :].T
        )
        return signal


class shs_spectrum:
    def __init__(self, specs):
        self.specs = specs
        self.pad_factor = 0

    def process_signal(self, signal):
        """ "Take the FFT and generate the power spectrum (abs(iFFT(signal))"""
        data = np.pad(signal, self.pad_factor * 100)
        x = np.fft.ifft(np.fft.ifftshift(data))
        s = np.fft.fftshift(x)
        return np.abs(s[:, 0 : int(self.specs.DetNumPixX / 2)])

    def get_wavenumber_scale(self, N, opd_sample_spacing):
        if N % 2 == 0:
            n = np.arange(-int(N / 2), int(N / 2), 1)
        else:
            n = np.arange(-int((N - 1) / 2), int((N - 1) / 2 + 1), 1)
        return n / (N * opd_sample_spacing)


class spectral_response_correction:

    """Removes the spectral response using the correction obtained from the lab characterization."""

    def __init__(self):
        self.filter_shape = []

    def process_signal(self, signal):
        # TODO: this will need t o be optimized for each flight since the filter shifts with temperature.
        self.filter_shape = signal["filter_shape"].data
        signal["spectrum"].data = signal["spectrum"].data / signal["filter_shape"].data
        return signal


class pixel_response_correction:

    """Removes the spectral response using the correction obtained from the lab characterization."""

    def __init__(self):
        self.pixel_response = []

    def process_signal(self, signal):
        self.pixel_response = signal["pixel_response"].data
        signal["spectrum"].data = (
            signal["spectrum"].data / signal["pixel_response"].data
        )
        return signal


class abscal:

    """Removes the spectral response using the correction obtained from the lab characterization."""

    def __init__(self):
        self.abs_cal = []

    def process_signal(self, signal):
        self.abs_cal = signal["abs_cal"].data
        signal["spectrum"].data = signal["spectrum"].data * signal["abs_cal"].data
        return signal


class DC_Filter:

    """Removes the DC component assuming that there are potential low frequency brightness fluctuations in the interferograms"""

    def __init__(self):
        self.dc_filter = []

    def dc_filt_func(self, f, s, N=4):
        return (0.5 * (1 + np.cos(np.pi * f / s))) ** N

    def process_signal(self, signal: np.ndarray) -> np.ndarray:
        data_image = signal["image"].data
        # Remove nans if they are still present
        data_image[np.isnan(data_image)] = 1

        iGM_fft = np.fft.rfft(data_image, axis=1)

        F = np.zeros_like(iGM_fft[0, :])
        fx = np.fft.rfftfreq(np.shape(data_image)[1], d=0.0015)
        s = 230
        f = fx[fx < s]

        F[fx < s] = self.dc_filt_func(f, s, N=100)
        interf_filter = xr.DataArray(
            F,
            dims=["pixelcolumn"],
            coords={"pixelcolumn": np.arange(0, np.shape(iGM_fft)[1])},
        )

        interf_filter = interf_filter.rolling(
            pixelcolumn=5, center=True, min_periods=1
        ).mean()

        # store the dc filter
        self.dc_filter = interf_filter.data

        S_filt = interf_filter.data * iGM_fft

        iGM_filt = np.fft.irfft(S_filt)
        scale = iGM_filt[:, 248]
        dc_correction = scale * (data_image / iGM_filt).T
        signal["image"].data = (dc_correction - np.mean(dc_correction, axis=0)).T
        return signal


class get_phase_corrected_spectrum:

    """Removes the spectral response using the correction obtained from the lab characterization."""

    def __init__(self):
        self.wavelengths = []
        self.specs = []

    def box_car(self, N, w, c):
        box = np.zeros(N)
        box[c - w : c + 1 + w] = 1
        return box

    def minfunc(self, a, freq, S_meas):
        wavL = self.specs["Littrow"]
        kL = 8 * np.pi * np.tan(self.specs["ThetaL"]) * wavL
        return np.abs(
            np.imag(S_meas) * np.cos(a[0] * (-freq + kL) + a[1])
            + np.real(S_meas) * np.sin(a[0] * (-freq + kL) + a[1])
        )
        # b = a[1]/kL + a[0]
        # return np.abs(np.imag(S_meas*np.exp(1j*(-a[0]*freq + a[0]*kL  + a[1]))))

    def phase(self, signal, freq, deg):
        return np.polyfit(freq[:], np.unwrap(np.angle(signal[:])), deg=deg)

    def get_full_spectrum(self, sample_interval, data: np.ndarray, pad_factor):
        """ "Take the FFT of each row in the image and return the full complex spectrum and spatial frequency scale

        Parameters
        ----------
        sample_interval: float
            sample spacing
        data: np.ndarray
            array containing the interferogram image
        Returns
        -------
        s: np.ndarray
            complex spectrum
        f: np.ndarray
            spatial frequencies

        """

        data = np.pad(data, pad_factor * 1000)
        N = len(data)
        delta = sample_interval

        x = np.fft.ifft(np.fft.ifftshift(data))
        s = np.fft.fftshift(x)

        if N % 2 == 0:
            n = np.arange(-int(N / 2), int(N / 2), 1)
        else:
            # n = np.arange(-int((N - 1) / 2), int((N - 1) / 2), 1)
            n = np.arange(-int((N - 1) / 2), int((N - 1) / 2 + 1), 1)
        f = n / (N * delta)
        return s, f

    def phase_correction(self, iGM):
        # make a box car function centered at the center of the signal with a width that is half the number of samples
        # slide the box car along the interferogram near the known center to find the location that minimizes the imaginary component
        array_center = int(len(iGM) / 2)
        d = []
        idx_list = np.arange(-5, 5)
        fft = []
        phase = []
        igms = []
        for i in idx_list:
            b = self.box_car(len(iGM), int(len(iGM) / 2) - 10, c=array_center + i)
            iGMc = b * iGM
            iGM_padded = []
            # pad the edge of the interferogram with zeros to keep it symmetric
            if i < 0:
                iGM_padded.append(np.pad(iGMc, (2 * np.abs(i), 0)))

            if i == 0:
                iGM_padded.append(iGMc)

            if i > 0:
                iGM_padded.append(np.pad(iGMc, (0, 2 * i)))

            igms.append(iGM_padded)
            iGM_fft, f_fft = self.get_full_spectrum(
                self.specs["opd_spacing"], data=iGM_padded[0], pad_factor=0
            )
            d.append(np.mean(np.diff(np.unwrap(np.angle(iGM_fft[100:160])))))
            phase.append(np.unwrap(np.angle(iGM_fft[100:160])))
            fft.append(iGM_fft)

        idx_center = np.argmin(np.abs(d))
        iGM_fft_center = fft[idx_center]
        n = len(iGM_fft_center)
        fx = (2 * np.pi / (n * np.mean(np.diff(self.specs["pos_x"])))) * np.arange(
            -(n - 1) / 2, (n - 1) / 2 + 1
        )
        kL = 8 * np.pi * np.tan(self.specs["ThetaL"]) * self.specs["Littrow"]

        lstsq_fit = least_squares(
            self.minfunc,
            (1e-05, 1),
            args=(fx[120:160], iGM_fft_center[120:160]),
            method="lm",
        )

        wav_cor = (fx + kL) / (8 * np.pi * np.tan(self.specs["ThetaL"]))

        corrected_spectrum = (
            iGM_fft_center
            * np.exp(-1j * fx * lstsq_fit.x[0])
            * np.exp(1j * kL * lstsq_fit.x[0])
            * np.exp(1j * lstsq_fit.x[1])
        )
        peak_idx = np.argmax(np.abs(corrected_spectrum[0:200]))

        if np.real(corrected_spectrum[peak_idx]) < 0:
            corrected_spectrum = corrected_spectrum * np.exp(1j * np.pi)

        # interpolate to a fixed wavenumber grid

        corrected_spectrum_fixed = np.interp(
            self.specs["wav_num"], wav_cor, corrected_spectrum
        )

        return (corrected_spectrum_fixed, self.specs["wav_num"])

    def process_signal(self, signal):
        iGM_cor = signal["image"].data

        specs = {}
        specs["DetNumPixX"] = len(signal["pixelcolumn"].data)
        specs["Littrow"] = signal["Littrow"].data
        specs["ThetaL"] = signal["ThetaL"].data
        specs["opd_spacing"] = np.mean(np.diff(signal["opd_x"].data))
        specs["pos_x"] = signal["pos_x"].data
        specs["wav_num"] = signal["wavenumbers"].data
        self.specs = specs

        CorSpec = []
        wavnums_list = []
        corrected_spectrum = 0
        wavnums = 0
        for i in range(np.shape(iGM_cor)[0]):
            try:
                corrected_spectrum, wavnums = self.phase_correction(iGM_cor[i, :])

            except ValueError:
                corrected_spectrum = np.ones_like(specs["wav_num"]) * np.nan
            CorSpec.append(corrected_spectrum)
            wavnums_list.append(wavnums)
        spectrum = np.reshape(
            np.concatenate(CorSpec), (len(CorSpec), np.shape(CorSpec[0])[0])
        ).T
        signal["spectrum"] = xr.DataArray(spectrum.T, dims=["los", "wavenumber"])
        return signal
