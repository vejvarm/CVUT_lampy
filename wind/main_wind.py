import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat

from wind.helpers import wd_str2rad
from bin.Preprocessor import Preprocessor


def load_wind(path):
    dct = loadmat(path)

    return np.array([d[0] for d in dct["WindDirection"].flatten()]), dct["WindSpeed"].flatten()


if __name__ == '__main__':
    path = "../data/serialized/20190425_Acc.mat"

    wdirr, wspeed = load_wind(path)

    p = Preprocessor()

    freqs, psd = p.run([path], return_as="ndarray")

    psd_mean = psd.mean(axis=(0, 1, 2))

    print(psd_mean.shape)

    for wdr, wspd, ps in zip(wdirr, wspeed, psd_mean.T):
        wdr_rad = wd_str2rad(wdr)
        plt.polar(wdr_rad, wspd+0.5, "xr")
        plt.text(wdr_rad, wspd+0.5, f"{int(ps*1000):2d}", fontsize=10)

    plt.show()
