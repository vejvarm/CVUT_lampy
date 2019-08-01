# normalizace -- calc_zscore
# fft (Nfft = 5120) -- calc_psd
# odstranění druhé poloviny spektra (zrcadlově symetrické s první polovinou)
# TODO: odstranění násobků 50 Hz (el.mag. rušení)
# TODO: vyhladit plovoucím průměrem (coarse graining) --- průměr z okolních hodnot
# TODO: odstranění trendů (0 - 50 Hz, 0 - 100 Hz)


import numpy as np
from scipy.io import loadmat


def mat2dict(path):
    """
    :param path: (str) path to .mat file with the data
    :return: (dict) structure with the files

    expected structure of data:
    :key 'Acc1': 2D float array [15360, 144] (1st lamp, 1st direction)
    :key 'Acc2': 2D float array [15360, 144] (1st lamp, 2nd direction)
    ...
    :key 'Acc6': 2D float array [15360, 144] (3rd lamp, 2nd direction)
    :key 'FrekvenceSignalu': int 512 Hz
    :key 'WindDirection': 1D string array [144] (dir. of wind described by characters N, S, E, W)
    :key 'WindSpeed': 1D float array [144] (speed of the wind)
    """
    return loadmat(path)


def calc_zscore(arr):
    """ calculate zscore normalized array from arr across the 0th dimension

    :param arr: 2D array to be normalized (expected dimensions: [15360, :])
    :return zscore of arr [15360, :]
    """
    assert len(arr.shape) == 2, "arr must be 2D array of [time samples, number of measurements]"
    return (arr - np.mean(arr, 0)) / np.std(arr, 0)


def calc_psd(arr, fs=512, ns_per_hz=10):
    # noinspection PyUnresolvedReferences
    """ calculate power spectral density from arr across the 0th dimension

    :param arr: 2D array of time signals to be FFTed and PSDed (expected dimensions: [15360, :])
    :param fs: sampling frequency of the signals in array (expected: 512 Hz)
    :param ns_per_hz: number of samples per one Hz in the returned psd (default: 10)
    :return: FIRST HALF OF THE freq_vals and psd (rest is redundant)
        :var freq_vals: 1D array of frequency values (x axis) [fs*ns_per_hz//2],
        :var psd: 2D array of power spectral densities of singals in arr [fs*ns_per_hz//2, :]
    """

    assert len(arr.shape) == 2, "arr must be 2D array of [time samples, number of measurements]"

    nfft = fs * ns_per_hz
    arr_fft = np.fft.fft(arr, nfft, 0)
    arr_psd = np.abs(arr_fft)**2 / nfft

    freq_vals = np.arange(0, fs, 1.0 / ns_per_hz)

    return freq_vals[0:nfft//2+1], arr_psd[0:nfft//2+1, :]


def remove_noise(psd_arr, fs=512, f_rem=tuple(range(50, 251, 50)), df_rem=(5, )*5):
    """

    :param psd_arr: array of power spectral densities [nfft, :]
    :param fs: sampling frequency of psd_arr (expected: 512 Hz)
    :param f_rem: 1D tuple of frequencies in Hz which should be removed from psd_arr
    :param df_rem: 1D tuple of distance which marks area around f_rem that should also be removed
    :return: denoised_psd_arr
    """

    nfft = psd_arr.shape[0]*2
    ns_per_hz = nfft/fs

    assert len(f_rem) == len(df_rem), "f_rem and df_rem must have the same number of values"

    for fr, dfr in zip(f_rem, df_rem):
        fr_idx = int(ns_per_hz*fr)
        print(fr_idx)
        dfr_idx = int(ns_per_hz*dfr)
        psd_arr[fr_idx-dfr_idx:fr_idx+dfr_idx+1] = 0

    return psd_arr
