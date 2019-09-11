import numpy as np


class Average:
    """
    Counting and keeping moving average of spectral values
    """
    def __init__(self, shape=(2561, )):
        """

        :param shape: dimension shapes of the psd entries (e.g. (n samples in fft / 2, ))
        """
        self.k = 0
        self.PSD = np.zeros(shape)

    def update(self, psd):
        self.k += 1
        self.PSD = (self.PSD * (self.k - 1) + psd) / self.k

    def reset(self):
        self.k = 0
        self.PSD = np.zeros(self.PSD.shape)


def complete_average(psd_arr):
    """
    Calculate average power spectral density value for each frequency from all given measurements in psd_arr

    :param psd_arr: array of power spectral densities [nfft, :]
    :return psd_arr_avg: average power spectral density from all given measurements
    """
    return np.mean(psd_arr, axis=1)
