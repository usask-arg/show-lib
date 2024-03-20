from __future__ import annotations

import os

import numpy as np
import xarray as xr
import yaml


class SHOW_specs:
    """
    This class reads in the specifications.yaml file and assigns the SHS specifications.
    Note: In this class it is assumed that all wavelengths are in air.

    """

    def __init__(self, littrow_nm=1363.38, M=0.21910042822432865, vac=False):
        # Littrow given in air. wavenumber scale and Littrow will be converterd to vacuuum if vac = True
        real_path = os.path.realpath(__file__)
        folder = real_path + r"\.."
        filename = folder + r"\er2_2023.yaml"
        self.vac = vac

        with open(filename) as stream:
            try:
                shs_specs_dict = yaml.full_load(stream)["shsdesign"]

            except:
                shs_specs_dict = yaml.load(stream)["shsdesign"]

        with open(filename) as stream:
            try:
                cbd_dict = yaml.full_load(stream)["cdb"]

            except:
                cbd_dict = yaml.load(stream)["cdb"]

        with open(filename) as stream:
            try:
                iFOV_dict = yaml.full_load(stream)["iFOV"]

            except:
                iFOV_dict = yaml.load(stream)["iFOV"]

        with open(filename) as stream:
            try:
                bad_pixels_dict = yaml.full_load(stream)["badpixels"]

            except:
                bad_pixels_dict = yaml.load(stream)["badpixels"]

        # Add all parameters to the design specifications:
        self.design = shs_specs_dict
        self.design.update(cbd_dict)
        self.design.update(iFOV_dict)
        self.design.update(bad_pixels_dict)
        self.GratingEdgeBottom = 147
        self.GratingEdgeTop = 588

        self.ImageEdgeLeft = self.design["Pixel_column_at_left_edge_of_grating_image"]
        self.ImageEdgeRight = self.design["Pixel_column_at_right_edge_of_grating_image"]
        self.ImageEdgeBottom = self.design["Pixel_row_at_bottom_of_atmospheric_image"]
        self.ImageEdgeTop = self.design["Pixel_row_at_top_edge_of_atmospheric_image"]
        self.BadPixelsSmall = self.design["small"]
        self.BadPixelsBig = self.design["big"]

        self.hfov = self.design["hfov"]
        self.vfov = -self.design["height_degrees_per_pixel"] * (
            self.ImageEdgeTop - self.ImageEdgeBottom
        )

        self.DetNumPixX = (
            self.design["Pixel_column_at_right_edge_of_grating_image"]
            - self.design["Pixel_column_at_left_edge_of_grating_image"]
        )
        self.DetNumPixY = (
            self.design["Pixel_row_at_top_edge_of_atmospheric_image"]
            - self.design["Pixel_row_at_bottom_of_atmospheric_image"]
        )

        self.DetRows = self.design["Number_of_rows_on_detector"]
        self.DetCols = self.design["Number_of_columns_on_detector"]

        self.DetWidthPixX = self.design["spectralpixel_width_cms"]
        self.DetWidthPixY = self.design["heightpixel_width_cms"]
        self.WidthDetX = self.DetWidthPixX * self.DetNumPixX
        self.WidthDetY = self.DetWidthPixY * self.DetNumPixY

        self.DetectorADU = self.design["electrons_per_DN"]
        self.DetectorReadout = self.design["detector_readout_noise"]
        self.DetectorSaturationLevel = self.design["dn_saturated_value"]

        self.bore_sight_pixel = self.design["bore_sight_pixel"]
        self.height_degrees_per_pixel = self.design["height_degrees_per_pixel"]
        self.boresight_pitchangle = self.design["boresight_pitchangle"]

        # Center Pixel
        self.CenterPixelHorizontal = self.design[
            "Centre_pixel_column_corresponding_to_x=0_in_the_grating_localization_plane"
        ]
        self.CenterPixelVertical = self.design[
            "Centre_pixel_in_vertical-spatial_dimension"
        ]

        # Dark Current Area
        self.DarkCurrentAreaBottomEdge = self.design[
            "Pixel_row_at_bottom_edge_of_dark_current_area_below_image_area"
        ]
        self.DarkCurrentAreaTopEdge = self.design[
            "0_pixel_row_at_top_edge_of_dark_current_area_below_image_area"
        ]

        # Magnification
        self.Magnification = M
        self.Littrow = 1e7 / littrow_nm

        # Littrow wavenumber
        self.SigmaL = self.Littrow  # [cm^-1]

        # Littrow angle
        self.ThetaL = np.arcsin(
            self.design["grating_groove_density_percm"] / (2 * self.SigmaL)
        )  # [rad]

        # Grating dimensions
        self.GratWidthX = (
            self.DetWidthPixX
            * (self.ImageEdgeRight - self.ImageEdgeLeft)
            * (1 / self.Magnification)
            / np.cos(self.ThetaL)
        )
        self.GratWidthY = (
            self.DetWidthPixY
            * (self.ImageEdgeTop - self.ImageEdgeBottom)
            * (1 / self.Magnification)
            / np.cos(self.ThetaL)
        )

        self.DSigma = (self.DetNumPixX / 2) / (
            4 * ((self.WidthDetX) * (1 / self.Magnification)) * np.tan(self.ThetaL)
        )  # [cm^-1]

        # Maximum observable spatial frequency
        self.FreqMax = 4 * np.tan(self.ThetaL) * self.DSigma

        self.min_wavel = 1e7 / (self.SigmaL + self.DSigma)
        self.max_wavel = 1e7 / (self.SigmaL - self.DSigma)

        self.PixelWidthXProj = self.DetWidthPixX / self.Magnification
        self.PixelWidthYProj = self.DetWidthPixY / self.Magnification

        # y dispersion
        self.Alpha = -5.079e-05  # [rad]
        self.GratOffDispRotRow = 1

        self.min_wavel = 1e7 / (self.SigmaL + self.DSigma)
        self.max_wavel = 1e7 / (self.SigmaL - self.DSigma)

        self.__SHS_optical_path__()
        self.__SHS_wavels__()
        self.__finite_pixel_response__()
        # self.__materion_filter__()
        self.__filter_correction__()
        self.__analytic_ILS__()

    def vac_to_air_wavels_nm(self, wavels_nm):
        wavels_um = wavels_nm * 1e-3
        n_air = (
            1
            + 0.05792105 / (238.0185 - wavels_um**-2)
            + 0.00167917 / (57.362 - wavels_um**-2)
        )
        return wavels_nm / n_air

    def air_to_vac_wavels_nm(self, wavels_nm):
        wavels_um = wavels_nm * 1e-3
        n_air = (
            1
            + 0.05792105 / (238.0185 - wavels_um**-2)
            + 0.00167917 / (57.362 - wavels_um**-2)
        )
        return wavels_nm * n_air

    def swap_wvnum_wvlen(self, in_value):
        """swaps wavenumber (cm-1) with wavelength (nm)"""
        return 1e7 / in_value

    def gaussian(self, x, amplitude, mean, stddev):
        return amplitude * np.exp(-0.5 * (((x - mean) / stddev) ** 2))

    def __analytic_ILS__(self, apodization=True):
        wav_target = 1e7 / 1366
        wav = np.arange(self.Littrow - self.FreqMax, self.Littrow + self.FreqMax, 0.001)
        u_p = 4 * np.tan(self.ThetaL) * (wav - self.Littrow)
        u = 4 * np.tan(self.ThetaL) * (wav_target - self.Littrow)
        xmax = np.max(self.pos_x)
        show_ils = (
            2 * xmax * (np.sinc(2 * xmax * (u_p - u)))
        )  # + np.sinc(2*xmax*(u_p+u)))
        wv_interp = np.arange(-10, 10 + 0.001, 0.001)
        ils = np.interp(wv_interp, wav - wav_target, show_ils / np.max(show_ils))

        L = self.max_opd
        apo = np.sinc(2 * L * wv_interp) / (1 - (2 * L * wv_interp) ** 2)
        if apodization is True:
            # ils = np.convolve(ils/np.sum(ils), apo, mode='same')
            amplitude = 1
            mean = 0
            stddev = 0.05
            w = self.gaussian(wv_interp, amplitude, mean, stddev)
            apo_f = np.convolve(w, apo, mode="same")
            apo_f = apo_f / np.max(apo_f)
            ils_f = np.convolve(apo_f, ils, mode="same")
            ils_f = ils_f / np.max(ils_f)

            self.show_analytic_ils = [wv_interp, ils_f]
        else:
            amplitude = 1
            mean = 0
            stddev = 0.072
            w = self.gaussian(wv_interp, amplitude, mean, stddev)
            ils_f = np.convolve(w / np.sum(w), ils, mode="same")
            self.show_analytic_ils = [wv_interp, ils_f]

    def __SHS_optical_path__(self):
        """Generates the optical path difference introduced between the two beams of the interferometer"""

        # Set the range of pixels
        self.interferogram_x_pixel = (
            np.arange(0, self.DetNumPixX) - self.CenterPixelHorizontal
        )
        self.interferogram_y_pixel = (
            np.arange(0, self.DetNumPixY) - self.CenterPixelVertical
        )

        # Calculate the position in cm on the grating
        self.pos_x = self.interferogram_x_pixel * self.PixelWidthXProj
        self.pos_y = self.interferogram_y_pixel * self.PixelWidthYProj

        # Calcualte the position in cm on the detector
        self.det_pos_x = self.interferogram_x_pixel * self.DetWidthPixX
        self.det_pos_y = self.interferogram_y_pixel * self.DetWidthPixY

        # Calculate the OPD.
        self.opd_x = 4 * np.tan(self.ThetaL) * self.pos_x
        self.opd_y = self.Alpha * self.pos_y
        self.max_opd = np.max(self.opd_x)
        self.opd_sampling = np.mean(np.diff(self.opd_x))

    def __SHS_wavels__(self):
        """Calculates the range of wavenumbers and wavelengths that can be observed with the SHS"""

        N = self.DetNumPixX
        delta = np.mean(np.diff(self.opd_x))
        if N % 2 == 0:
            n = np.arange(-int(N / 2), int(N / 2), 1)
        else:
            n = np.arange(-int((N - 1) / 2), int((N - 1) / 2 + 1), 1)

        f = n / ((N) * delta)
        self.spatial_freq = f
        wav_num = f + self.SigmaL
        self.wav_num = wav_num
        self.spectrum_wavelengths = np.flip(1e7 / self.wav_num)
        if self.vac is True:
            self.spectrum_wavelengths = self.air_wavelength_to_vacuum_wavelength(
                self.spectrum_wavelengths
            )
            self.wav_num = np.flip(1e7 / self.spectrum_wavelengths)
        self.wavenumber_spacing = np.mean(np.diff(self.wav_num))
        self.opd_delta = delta

    def __finite_pixel_response__(self):
        self.pixel_response = np.sinc(
            (self.wav_num - self.Littrow)
            * 4
            * np.tan(self.ThetaL)
            * self.DetWidthPixX
            / self.Magnification
        )

    def __materion_filter__(self):
        path = r"C:\Users\jeffl\SHOWER2_2023\data\materion_transmission_profiles.txt"

        trans = np.flip(np.loadtxt(path)[:, 1] / 100)
        full_trans = np.flip(np.loadtxt(path)[:, 1::] / 100)
        # The SHOW materion filter was designed with a center wavelength of 1366.86 nm. However, characterization measurements performed at DRDC suggest that the center
        # is shifted down by 0.3 nm to 1366.56 nm
        wav_filter = np.flip(1e7 / (np.loadtxt(path)[:, 0]))

        self.trans_filter = np.interp(self.wav_num, wav_filter, trans)
        self.materion_transmittance = []
        for i in range(np.shape(full_trans)[1]):
            self.materion_transmittance.append(
                np.interp(self.wav_num, wav_filter, full_trans[:, i])
            )

    def __filter_correction__(self):
        real_path = os.path.realpath(__file__)
        folder = real_path + r"\.."
        path_abscal = folder + r"\SHOW_ER2_abscal_2024_02_08_v0.2.nc"
        abscal_data = xr.open_dataset(path_abscal)

        # path = r'show_example_for_zawada.nc'
        # filter_cor = xr.open_dataset(path)
        # self.spectral_response_wavenumbers = self.wav_num[0:247]
        # self.spectral_response = np.interp(self.wav_num[0:247], filter_cor.wavelength.data, filter_cor.filter_correction.data)
        self.spectral_response_wavenumbers = self.wav_num[0:247]
        self.spectral_response = np.interp(
            self.wav_num[0:247],
            abscal_data.wavenumbers.data,
            abscal_data.filter_shape.data,
        )

        self.abs_cal = 0.475 * abscal_data.abscal.data

    def air_wavelength_to_vacuum_wavelength(self, wavelength_nm: np.array) -> np.array:
        """
        Converts wavelength specified in Air at STP to wavelengths at vacuum

        Parameters
        ----------
        wavelength_nm : np.array
            Wavelength in air at STP

        Returns
        -------
        np.array
            Vacuum wavelengths
        """
        s = 1e4 / (wavelength_nm * 10)  # Convert to angstroms

        refac_index = (
            1
            + 0.00008336624212083
            + 0.02408926869968 / (130.1065924522 - s**2)
            + 0.0001599740894897 / (38.92568793293 - s**2)
        )

        return wavelength_nm * refac_index

    def vacuum_wavelength_to_air_wavelength(wavelength_nm: np.array) -> np.array:
        """
        Converts wavelength specified in vacuum to wavelengths at air at STP

        Parameters
        ----------
        wavelength_nm : np.array
            Wavelengths in vacuum

        Returns
        -------
        np.array
            Wavelengths in air at STP
        """
        s = 1e4 / (wavelength_nm * 10)  # Convert to angstroms

        n = (
            1
            + 0.0000834254
            + 0.02406147 / (130 - s**2)
            + 0.00015998 / (38.9 - s**2)
        )

        return wavelength_nm / n


if __name__ == "__main__":
    specs = SHOW_specs()
    print("done")
