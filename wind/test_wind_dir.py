import unittest
import numpy as np

from wind.helpers import wd_str2rad


class MyTestCase(unittest.TestCase):
    ss = ["E", "ENE", "NE", "NNE", "N", "NNW", "NW", "WNW", "W", "WSW", "SW", "SSW", "S", "SSE", "SE", "ESE", "E"]
    ss_rad_true = np.linspace(0, 2, 17)*np.pi
    ss_rad_true[-1] = 0.
    x = np.linspace(1, 10, len(ss))

    def test_mapping(self):
        thetas = np.array([wd_str2rad(s) for s in self.ss])

        np.testing.assert_equal(thetas, self.ss_rad_true, "Mapping did not work as expected!")



if __name__ == '__main__':
    unittest.main()
