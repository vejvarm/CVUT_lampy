# normalizace -- calc_zscore
# hammingovo okno před FFT
# fft (Nfft = 5120) -- calc_psd
# odstranění druhé poloviny spektra (zrcadlově symetrické s první polovinou)
# odstranění násobků 50 Hz (el.mag. rušení)
# vyhladit plovoucím průměrem (coarse graining) --- průměr z okolních hodnot
# odstranění trendů (0 - 50 Hz, 0 - 100 Hz)


import os
from copy import deepcopy

import numpy as np
from scipy.io import loadmat
from scipy.optimize import leastsq
from tqdm import tqdm


def _remove_negative(psd_arr):
    """
    Remove negative values from psd

    :param psd_arr: array of power spectral densities [nfft, :]
    :return: psd_arr_positive
    """
    return psd_arr.clip(min=0)


class Preprocessor:

    def __init__(self, fs=512, ns_per_hz=10, freq_range=(0, 256), noise_f_rem=tuple(range(50, 251, 50)), noise_df_rem=(5, )*5, mov_filt_size=5):
        """

        :param fs: (int) sampling frequency of the acquisition
        :param ns_per_hz: (int) desired number of samples per Hertz in FFT
        :param freq_range: (list/tuple) (minimum frequency, maximum frequency) rest is thrown away
        :param noise_f_rem: (list/tuple) frequencies that should be removed (zeroed out) from the power spectrum
        :param noise_df_rem: (list/tuple) range around f_rem that should also be removed
        :param mov_filt_size: (int) length of the rectangular filter for moving average application
        """
        self.fs = fs
        self.ns_per_hz = ns_per_hz
        self.freq_range = freq_range
        self.noise_f_rem = noise_f_rem
        self.noise_df_rem = noise_df_rem
        self.mov_filt_size = mov_filt_size

    def run(self, paths):
        """

        :param paths: (list/tuple) strings of paths leading to .mat files or folder with .mat files for loading
        :return preprocessed: (Dict[file_name: Tuple[freq, psd, wind_dir, wind_spd]]) dictionary of preprocessed files
        """

        preprocessed = dict()

        for path in paths:
            path = os.path.normpath(path)  # normalize the path structure
            if os.path.isdir(path):
                # leads to folder ... load all .mat files from it
                path_gen = os.walk(path)
                for p, sub, files in path_gen:
                    with tqdm(desc=f"Processing files in folder {p}", total=len(files), unit="file") as pbar:
                        for file in files:
                            f_name, f_ext = os.path.splitext(file)
                            if f_ext == '.mat':
                                fullpath = os.path.join(p, file)
                                freq_vals, psd_list, wind_dir, wind_spd = self._preprocess(fullpath)
                                preprocessed[f_name] = (freq_vals, psd_list, wind_dir, wind_spd)
                            pbar.update(1)

            elif os.path.splitext(path)[-1] == '.mat':
                # leads to .mat file ... load directly
                freq_vals, psd_list, wind_dir, wind_spd = self._preprocess(path)
                f_name = os.path.splitext(os.path.basename(path))[0]
                preprocessed[f_name] = (freq_vals, psd_list, wind_dir, wind_spd)
            else:
                print("Given path doesn't refer to .mat file or folder with .mat files. \n Ignoring path.")
                continue

        return preprocessed

    # TODO: load all files from paths into dicts
    # TODO: preprocess each dict sepparately or all into one?

    def _preprocess(self, fullpath2file):
        """ run the preprocessing pipeline

        :param fullpath2file: full path to .mat file that should be preprocessed
        :return:
            :var freq_vals:  (1D ndarray) frequency values (x axis) [fs*ns_per_hz//2],
            :var psd_list: (List[2D ndarray]) preprocessed power spectral densities [self.nfft, number of measurements]
            :var wind_dir: (1D string array) [144] (dir. of wind described by characters N, S, E, W)
            :var wind_spd: (1D float array) [144] (speed of the wind)
        """
        df = self._mat2dict(fullpath2file)

        acc = [[]]*6
        freq_vals = [[]]*len(acc)
        psd_list = [[]]*len(acc)

        acc[0] = df['Acc1']
        acc[1] = df['Acc2']
        acc[2] = df['Acc3']
        acc[3] = df['Acc4']
        acc[4] = df['Acc5']
        acc[5] = df['Acc6']

        # check if self.fs fits to value of 'FrekvenceSignalu'
        assert df['FrekvenceSignalu'] == self.fs, f"Value of 'FrekvenceSignalu' in {fullpath2file} doesn't correspond with self.fs ({self.fs} Hz)"

        for i, a in enumerate(acc):
            # normalize 'a' to 0 mean and 1 variance
            a = self._calc_zscore(a)
            # calculate frequency values and power spectral density (psd) of 'a'
            freq_vals[i], psd = self._calc_psd(a)
            # remove noise based on values in self.noise_f_rem and self.noise_df_rem
            psd = self._remove_noise(psd, replace_with="zero")
            # use moving average to smooth out the spectrum and remove noise
            psd = self._coarse_grain(psd)
            # normalize to 0 mean and 1 var
            psd_mean = np.mean(psd)
            psd_std = np.std(psd)
#            psd = (psd - psd.min())/(psd.max() - psd.min())
            psd = (psd - psd_mean)/psd_std
#            print(np.mean(psd), np.var(psd), psd.max())
            # remove trend
            psd = self._detrend(freq_vals[i], psd)
            # remove negative values
            psd = self._remove_negative(psd)
            # remove everything lower than mode:
#            psd = self._remove_below_mode(psd)
            # save to list of psd for each acceleration sensor output
            psd_list[i] = psd

        # check if freq_vals are generated the same in every instance of psd
        for i in range(len(freq_vals) - 1):
            np.testing.assert_array_equal(freq_vals[i], freq_vals[i+1], 'Frequency values are not consistent')

        wind_dir = df['WindDirection'] if 'WindDirection' in df.keys() else None
        wind_spd = df['WindSpeed'] if 'WindSpeed' in df.keys() else None

        return freq_vals[0], psd_list, wind_dir, wind_spd

    @staticmethod
    def _mat2dict(path):
        """
        :param path: (str) path to .mat file with the data
        :return: (dict) structure with the files

        expected structure of data:
        :key 'Acc1': 2D float array [15360, 144] (1st lamp, 1st direction)
        :key 'Acc2': 2D float array [15360, 144] (1st lamp, 2nd direction)
        ...
        :key 'Acc6': 2D float array [15360, 144] (3rd lamp, 2nd direction)
        :key 'FrekvenceSignalu': 1D uint16 array [1] (512 Hz)
        :key 'WindDirection': 1D string array [144] (dir. of wind described by characters N, S, E, W)
        :key 'WindSpeed': 1D float array [144] (speed of the wind)
        """
        return loadmat(path)

    @staticmethod
    def _calc_zscore(arr):
        """ calculate zscore normalized array from arr across the 0th dimension

        :param arr: 2D array to be normalized (expected dimensions: [15360, :])
        :return zscore of arr [15360, :]:
        """
        assert len(arr.shape) == 2, "arr must be 2D array of [time samples, number of measurements]"
        return (arr - np.mean(arr, 0)) / np.std(arr, 0)

    @staticmethod
    def _hamming(arr):
        """Apply Hamming window across the 0th dimension in arr"""
        return (arr.T*np.hamming(arr.shape[0])).T

    def _calc_psd(self, arr):
        """ calculate power spectral density from arr across the 0th dimension

        :param arr: 2D array of time signals to be FFTed and PSDed (expected dimensions: [15360, :])
        :return: FIRST HALF OF THE freq_vals and psd (rest is redundant)
            :var freq_vals: 1D array of frequency values in self.freq_range (x axis) [fs*ns_per_hz//2],
            :var psd: 2D array of power spectral densities of singals in arr [fs*ns_per_hz//2, :]
        """
        assert len(arr.shape) == 2, "arr must be 2D array of [time samples, number of measurements]"

        nfft = self.fs * self.ns_per_hz
        arr = self._hamming(arr)
        arr_fft = np.fft.fft(arr, nfft, 0)
        arr_psd = np.abs(arr_fft)**2 / nfft

        freq_vals = np.arange(0, self.fs, 1.0 / self.ns_per_hz)

        # restrict frequency values to the range of self.freq_range
        freq_slice = slice(self.freq_range[0]*self.ns_per_hz, self.freq_range[1]*self.ns_per_hz)

        return freq_vals[freq_slice], arr_psd[freq_slice, :]

    def _remove_noise(self, psd_arr, replace_with="zero"):
        """

        :param psd_arr: array of power spectral densities [nfft, :]
        :param replace_with: (string) with what should the removed frequencies be replaced
            :val zero: set to 0 value
            :val min: set to min value in psd
            :val mean: set to mean value in psd
            :val repeat: repeat the last valid value in psd
        :return psd_arr_denoised: unwanted parts are replaced by constant value from the value directly before the removal
        """
        psd_arr_denoised = deepcopy(psd_arr)
        min_psd = np.min(psd_arr)
        mean_psd = np.mean(psd_arr)

        assert len(self.noise_f_rem) == len(self.noise_df_rem), "f_rem and df_rem must have the same number of values"

        for fr, dfr in zip(self.noise_f_rem, self.noise_df_rem):
            fr_idx = int(self.ns_per_hz*(fr - self.freq_range[0]))
            dfr_idx = int(self.ns_per_hz*dfr)
            if replace_with == "zero":
                psd_arr_denoised[fr_idx-dfr_idx:fr_idx+dfr_idx+1] = 0
            elif replace_with == "min":
                psd_arr_denoised[fr_idx-dfr_idx:fr_idx+dfr_idx+1] = min_psd
            elif replace_with == "mean":
                psd_arr_denoised[fr_idx - dfr_idx:fr_idx + dfr_idx + 1] = mean_psd
            elif replace_with == "repeat":
                psd_arr_denoised[fr_idx-dfr_idx:fr_idx+dfr_idx+1] = psd_arr[fr_idx-dfr_idx-1]
            else:
                raise AttributeError("replace_with param should be one of: 'zero' | 'min' | 'repeat'")

        return psd_arr_denoised

    def _coarse_grain(self, psd_arr):
        """
        Calculating moving average using rectangular filter of length "filt_len"

        :param psd_arr: array of power spectral densities [nfft, :]

        :return psd_arr_cg: coarse grained psd_arr using moving average
        """
        psd_arr_cg = np.zeros(psd_arr.shape)
        filt = np.ones(self.mov_filt_size) / self.mov_filt_size

        for i in range(psd_arr.shape[1]):
            psd_arr_cg[:, i] = np.convolve(psd_arr[:, i], filt, 'same')

        return psd_arr_cg

    def _remove_negative(self, psd_arr):
        """
        Remove negative values from psd

        :param psd_arr: array of power spectral densities [nfft, :]
        :return: psd_arr_positive
        """
        return psd_arr.clip(min=0)

    def _remove_below_mode(self, psd_arr):
        """
        Remove everything below mode

        :param psd_arr:
        :return: psd_arr_clipped
        """

        hist = np.histogram(psd_arr, 100)
        mode = hist[1][int(np.argmax(hist[0]))]

        return psd_arr.clip(min=mode)

    def _detrend(self, freqs, psd_arr):
        """
        Remove linear trend from y=psd_arr(x) based on Least Squares optimized curve fitting

        :param freqs: x
        :param psd_arr: y
        :return: psd_arr_detrended
        """
        nfs, nms = psd_arr.shape
        psd_arr_detrended = np.zeros((nfs, nms))
        min_idx, max_idx = (int(self.ns_per_hz*f) for f in self.freq_range)

        for i in range(nms):
            par_init = (1., )

            x = freqs[min_idx:max_idx]
            y = psd_arr[min_idx:max_idx, i]

            par, success = leastsq(self.__error_func, par_init, args=(x, y, self.__linear_func))

            y_trend = self.__linear_func(par, x)

            psd_arr_detrended[:, i] = np.concatenate((psd_arr[:min_idx, i], y - y_trend, psd_arr[max_idx:, i]))

        return psd_arr_detrended

    @staticmethod
    def __error_func(par, x, y, func):
        return np.square(y - func(par, x))

    @staticmethod
    def __linear_func(par, x):
        return np.sum(par[i]*x**i for i in range(len(par)))
