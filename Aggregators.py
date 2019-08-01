import numpy as np


class Average:
    """
    Counting and keeping moving average of spectral values
    """
    def __init__(self, psd):
        self.PSD = psd
        self.k = 1

    def update(self, psd):
        self.k += 1
        self.PSD = (self.PSD * (self.k - 1) + psd) / self.k

    def reset(self):
        self.PSD = np.zeros(self.PSD.shape)
        self.k = 0