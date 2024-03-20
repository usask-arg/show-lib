from __future__ import annotations

import numpy as np


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
        import copy

        test_image = copy.deepcopy(image)
        # These pixeld were identified in the L1B as either dust or bad pixels.

        p = (
            [206, 92],
            [154, 294],
            [154, 295],
            [155, 295],
            [114, 155],
            [129, 199],
            [133, 200],
            [108, 53],
            [108, 51],
            [76, 88],
            [31, 56],
            [129, 199],
            [133, 200],
            [27, 226],
            [24, 225],
            [209, 25],
            [180, 51],
            [44, 3],
            [146, 25],
            [90, 170],
            [85, 174],
            [41, 185],
            [45, 186],
            [150, 174],
            [129, 199],
            [130, 199],
            [172, 301],
            [81, 320],
            [184, 417],
            [113, 368],
            [113, 369],
            [80, 363],
            [80, 364],
            [74, 372],
            [83, 364],
            [105, 439],
            [190, 330],
            [6, 420],
            [180, 51],
            [181, 51],
            [12, 434],
            [178, 286],
            [249, 6],
            [302, 66],
            [35, 109],
            [74, 11],
            [78, 11],
            [35, 286],
            [25, 438],
            [271, 336],
            [11, 78],
            [35, 107],
            [274, 178],
            [269, 196],
            [238, 130],
            [219, 292],
            [96, 444],
            [154, 436],
            [194, 273],
            [167, 328],
            [156, 80],
            [61, 188],
            [296, 363],
            [296, 452],
            [155, 148],
            [76, 113],
            [91, 127],
            [7, 328],
            [225, 233],
            [187, 493],
            [187, 77],
            [137, 244],
            [214, 101],
        )

        # bigger dust particles
        pb1 = [197, 199, 143, 146]
        pb2 = [201, 202, 145, 147]
        pb3 = [206, 208, 3, 4]
        pb4 = [110, 112, 154, 155]
        pb5 = [157, 159, 295, 298]
        pb6 = [49, 51, 385, 388]
        pb7 = [52, 54, 386, 389]
        pb8 = [119, 121, 455, 457]
        pb9 = [122, 125, 456, 458]
        pb10 = [224, 231, 16, 20]

        for i in range(len(p)):
            image[tuple(p[i])] = np.nan

        image[180, 253] = np.nan
        image[172, :] = np.nan
        image[pb1[0] : pb1[1], pb1[2] : pb1[3]] = np.nan
        image[pb2[0] : pb2[1], pb2[2] : pb2[3]] = np.nan
        image[pb3[0] : pb3[1], pb3[2] : pb3[3]] = np.nan
        image[pb4[0] : pb4[1], pb4[2] : pb4[3]] = np.nan
        image[pb5[0] : pb5[1], pb5[2] : pb5[3]] = np.nan
        image[pb6[0] : pb6[1], pb6[2] : pb6[3]] = np.nan
        image[pb7[0] : pb7[1], pb7[2] : pb7[3]] = np.nan
        image[pb8[0] : pb8[1], pb8[2] : pb8[3]] = np.nan
        image[pb9[0] : pb9[1], pb9[2] : pb9[3]] = np.nan
        image[pb10[0] : pb10[1], pb10[2] : pb10[3]] = np.nan

        # import matplotlib.pyplot as plt
        # plt.figure(2, figsize=[10, 6])
        # plt.rcParams.update({'font.size': 15})
        # image[230::, :] = np.nan
        # plt.pcolormesh(image)
        # plt.title('L1A Image - Bad Pixel Map Applied')
        # plt.clim(-500, 1000)
        # plt.colorbar(label='[DN/s]')
        # plt.ylabel('row')
        # plt.xlabel('column')
        return image

    def process_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Process the signal
        :param signal: input signal
        :return: processed output signal
        """

        return self.interp_bad_pixels(signal)
