from .l1a import l1a_bad_pixel_removal as bad_pixels
from collections import OrderedDict

class level1A_processing(object):
    """
    Implements Level L1A processing of SHOW interferogram images
    """

    def __init__(self,specs):
        """
        :param specs:
        :param num_samples:
        :param pad_factor:
        """
        self.specs = specs
        self._level1_data_processing = OrderedDict()
        self.__create_L1_processors__()


    def __create_L1_processors__(self):
        """Creates a L1 processing element
        """
        self.add_component(bad_pixels(self.specs), 'bad_pixels')

    def add_component(self, processor, name):
        """ Adds a component to the processor"""
        self._level1_data_processing[name] = processor

    def process_signal(self, dataset):
        signal = dataset['image'].data
        process_array = []
        for name, processor in self._level1_data_processing.items():

            signal = processor.process_signal(signal)
            process_array.append(signal)

        dataset['image'].data = signal
        return dataset


