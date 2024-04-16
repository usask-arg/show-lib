from __future__ import annotations

import numpy as np
from scipy.signal import resample


class l1a_bad_pixel_removal:
    """
    Used to correct bad pixels"""

    def __init__(self, specs):
        self.specs = specs

    #
    def dead_pixels(self, image):
        ix0 = self.specs.ImageEdgeLeft
        iy0 = self.specs.ImageEdgeBottom
        for p in self.specs.BadPixelsSmall:
            # place nans at the bad pixel locations
            px = p[0] - ix0 - 1
            py = p[1] - iy0 - 1
            image[py, px] = np.nan

        image[
            self.specs.BadPixelsBig[0][2]
            - iy0
            - 1 : self.specs.BadPixelsBig[0][3]
            - iy0
            - 1,
            self.specs.BadPixelsBig[0][0]
            - ix0
            - 1 : self.specs.BadPixelsBig[0][1]
            - ix0
            - 1,
        ] = np.nan
        image[np.where(image == 16383)] = np.nan
        return image

    def interp_bad_pixels(self, image):
        """
        Interpolate bad pixels
        :param image: input image
        :return: output image with bad pixels interpolated

        """
        # image = self.nan_bad_pixels(image)
        # apply after identifying the dead/bad pixels
        image = self.dust_offset_pixels(image)
        image = self.dead_pixels(image)

        image[np.where(image == 16383)] = np.nan
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

    def dust_offset_pixels(self, image):
        return self.specs.dust_mask * image

    def process_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Process the signal
        :param signal: input signal
        :return: processed output signal
        """

        return self.interp_bad_pixels(signal)


class find_zpd:
    """
    Used to correct bad pixels"""

    def __init__(self, specs):
        self.specs = specs

    def zpd(self, iGM: np.ndarray) -> np.ndarray:
        """
        Process the signal
        :param signal: input signal
        :return: processed output signal
        """
        l1a_bad_pixel_removal(self.specs).interp_bad_pixels(iGM)

        # find the row corresponding to the zero tilt
        zero_tilt_row = np.argmin((np.abs(iGM[:, 245] - iGM[:, 253]))[110:200]) + 110
        grid = np.arange(0, len(iGM[zero_tilt_row, :]))

        resample_iGM, resampled_grid = resample(
            iGM[zero_tilt_row, :], 1000 * len(iGM[zero_tilt_row, :]), t=grid
        )
        return resampled_grid[np.argmax(resample_iGM)]
