import unittest

import numpy as np

from bin.Aggregators import Average


class TestAggregators(unittest.TestCase):

    def test_mov_average(self):
        """ test if moving average is being calculated correctly by Aggregators.Average class """
        n_freq = int(512/2)  # number of frequencies in one measurement
        n_meas = int(6*24)   # measurements per day
        d = 5           # decimal places for rounding

        psd = np.random.randn(n_freq, n_meas)
        cAvg = Average(psd[:, 0])

        for k in range(2, n_meas + 1):
            cAvg.update(psd[:, k-1])

        # Test shape
        self.assertEqual(psd.shape[0], cAvg.PSD.shape[0], "Shapes of PSD and psd should be consistent")

        # Test values
        PSD_r, mean_psd_r = np.round(cAvg.PSD, d), np.round(np.mean(psd, 1), d)
        np.testing.assert_array_equal(PSD_r, mean_psd_r, "Moving average is not consistent with overall average")


if __name__ == '__main__':
    unittest.main()
