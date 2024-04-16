from __future__ import annotations

from collections import OrderedDict

from .l1a import find_zpd as zpd
from .l1a import l1a_bad_pixel_removal as bad_pixels


class level1A_processing:
    """
    Implements Level L1A processing of SHOW interferogram images
    """

    def __init__(self, specs):
        """
        :param specs:
        :param num_samples:
        :param pad_factor:
        """
        self.specs = specs
        self._level1_data_processing = OrderedDict()
        self.__create_L1_processors__()

    def __create_L1_processors__(self):
        """Creates a L1 processing element"""
        self.add_component(bad_pixels(self.specs), "bad_pixels")

    def add_component(self, processor, name):
        """Adds a component to the processor"""
        self._level1_data_processing[name] = processor

    def process_signal(self, dataset):
        signal = dataset["image"].data
        noise = dataset["error"].data
        process_array = []
        for _name, processor in self._level1_data_processing.items():
            signal = processor.process_signal(signal)
            noise = processor.process_signal(noise)

            process_array.append(signal)

        dataset["image"].data = signal
        dataset["error"].data = noise

        zpd_loc = zpd(self.specs).zpd(signal["image"].data)
        dataset["zpd"] = zpd_loc

        return dataset
