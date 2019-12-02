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

from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.optimize import leastsq
from scipy.linalg import hankel
from tqdm import tqdm


def _remove_negative(psd_arr):
    """
    Remove negative values from psd

    :param psd_arr: array of power spectral densities [nfft, :]
    :return: psd_arr_positive
    """
    return psd_arr.clip(min=0)


class Preprocessor:

    def __init__(self,
                 fs=512,
                 ns_per_hz=10,
                 freq_range=(0, 256),
                 noise_f_rem=(2, 50, 100, 150, 200),
                 noise_df_rem=(2, 5, 2, 5, 2),
                 mov_filt_size=5):
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

    def get_config_values(self):
        return tuple(self.__dict__.values())

    def run(self, paths, return_as='dict', plot_semiresults=False, nmeas=None):
        """

        :param paths: (list/tuple) strings of paths leading to .mat files or folder with .mat files for loading
        :param return_as: (string) if 'dict', returns a dictionary, if 'ndarray' returns an ndarray of only psd
        :param plot_semiresults: if True, make a plot after each preprocessing operation
        :param nmeas: if None, take all measurements available in file, if (int), take first nmeas measurements
        :return preprocessed:
            if return_as == 'dict': Dict[file_name: Tuple[freq, psd, wind_dir, wind_spd]] dict of preprocessed files
            if return_as == 'ndarray': Tuple[1Darray[nfft/2, ], 4Darray[nfiles, naccs, nfft/2, nmeas]]
        """

        preprocessed = dict()

        # for .npy file path!
        if ".npy" in os.path.splitext(paths[0])[-1]:
            # leads to .npy file ... load directly
            folder_path = os.path.split(paths[0])[0]
            freqs = np.load(folder_path + "/freqs.npy")
            psd_stacked = np.load(paths[0])
            print(".npy path specified. Loading and returning only first file from paths!")
            return freqs, psd_stacked

        # for .mat or folder paths
        for path in paths:
            path = os.path.normpath(path)  # normalize the path structure
            if os.path.isdir(path):
                # leads to folder ... load all .mat files from it
                path_gen = os.walk(path)
                for p, sub, files in path_gen:
                    mat_files = [file for file in files if os.path.splitext(file)[-1] == ".mat"]
                    with tqdm(desc=f"Processing files in folder {p}", total=len(mat_files), unit="file") as pbar:
                        for file in mat_files:
                            f_name, f_ext = os.path.splitext(file)
                            if f_ext == '.mat':
                                fullpath = os.path.join(p, file)
                                freq_vals, psd_list, wind_dir, wind_spd = self._preprocess(fullpath, plot_semiresults, nmeas)
                                preprocessed[f_name] = (freq_vals, psd_list, wind_dir, wind_spd)
                            pbar.update(1)
            elif os.path.splitext(path)[-1] == '.mat':
                # leads to .mat file ... load directly
                freq_vals, psd_list, wind_dir, wind_spd = self._preprocess(path, plot_semiresults, nmeas)
                f_name = os.path.splitext(os.path.basename(path))[0]
                preprocessed[f_name] = (freq_vals, psd_list, wind_dir, wind_spd)
            else:
                print("Given path doesn't refer to .mat file or folder with .mat files. \n Ignoring path.")
                continue

        if return_as == 'dict':
            return preprocessed
        elif return_as == 'ndarray':
            for key, (freqs, psd_list, _, _) in preprocessed.items():
                psd_array = np.array(psd_list)
                if 'psd_stacked' in locals():
                    psd_stacked = np.concatenate((np.expand_dims(psd_array, 0), psd_stacked))
                else:
                    psd_stacked = np.expand_dims(psd_array, 0)
            return freqs, psd_stacked
        else:
            raise AttributeError("return_as should be either 'dict' or 'ndarray'")

    def _preprocess(self, fullpath2file, plot_semiresults=False, nmeas=None):
        """ run the preprocessing pipeline

        :param fullpath2file: full path to .mat file that should be preprocessed
        :param plot_semiresults: if True, make a plot after each preprocessing operation
        :param nmeas: if None, take all measurements available in file, if (int), take first nmeas measurements
        :return:
            :var freq_vals:  (1D ndarray) frequency values (x axis) [fs*ns_per_hz//2],
            :var psd_list: (List[2D ndarray]) preprocessed power spectral densities [self.nfft, number of measurements]
            :var wind_dir: (1D string array) [144] (dir. of wind described by characters N, S, E, W)
            :var wind_spd: (1D float array) [144] (speed of the wind)
        """
        df = self._mat2dict(fullpath2file)

        naccs = 6
        freq_vals = [[]]*naccs
        psd_list = [[]]*naccs

        acc = [df[f'Acc{i}'] for i in range(1, naccs + 1)]

        if nmeas:
            for i in range(naccs):
                acc[i] = acc[i][:, :nmeas]

        # check if self.fs fits to value of 'FrekvenceSignalu'
        assert df['FrekvenceSignalu'] == self.fs, f"Value of 'FrekvenceSignalu' in {fullpath2file} doesn't correspond with self.fs ({self.fs} Hz)"

        for i, a in enumerate(acc):

            nvals, nmeas = a.shape
            nops = 8  # number of preprocessing operations (for plotting)
            time = np.arange(nvals) / self.fs

            # initialize plot if 'plot_semiresults' == True
            if plot_semiresults:
                fig, ax = plt.subplots(nops+1, 1)
            else:
                ax = [None]*(nops+1)

            self.conditional_plot(time, a.mean(axis=1), ax[0], plot_semiresults, xlabel='time (s)', ylabel='μ(a)', title='raw acceleration (acc)')

            # normalize 'a' to mean==0 and variance==1
            a = self._calc_zscore(a)
            self.conditional_plot(time, a.mean(axis=1), ax[1], plot_semiresults, xlabel='time (s)', ylabel='μ(a)', title='normalized acc')

            # calculate autocorrelation function to reduce noise
            a = self._autocorr(a)
            self.conditional_plot(time[:-1], a.mean(axis=1), ax[2], plot_semiresults, xlabel='time (s)', ylabel='ρxx', title='autocorrelation of acc')

            # calculate frequency values and power spectral density (psd) of 'a'
            freq_vals[i], psd = self._calc_psd(a)
            self.conditional_plot(freq_vals[i], psd.mean(axis=1), ax[3], plot_semiresults, xlabel='freqency (Hz)', ylabel='psd', title='Power spectral density (psd)')

            # remove noise based on values in self.noise_f_rem and self.noise_df_rem
            psd = self._remove_noise(psd, replace_with="min")
            self.conditional_plot(freq_vals[i], psd.mean(axis=1), ax[4], plot_semiresults, xlabel='freqency (Hz)', ylabel='psd', title='psd without *50Hz')

            # use moving average to smooth out the spectrum and remove noise
            psd = self._coarse_grain(psd)
            self.conditional_plot(freq_vals[i], psd.mean(axis=1), ax[5], plot_semiresults, xlabel='freqency (Hz)', ylabel='psd', title='coarse grained psd')

            # normalize 'psd' to mean==0 and variance==1
            psd = (psd - psd.mean())/psd.std()
            self.conditional_plot(freq_vals[i], psd.mean(axis=1), ax[6], plot_semiresults, xlabel='freqency (Hz)', ylabel='psd', title='normalized psd')

            # remove trend
            psd = self._detrend(freq_vals[i], psd)
            self.conditional_plot(freq_vals[i], psd.mean(axis=1), ax[7], plot_semiresults, xlabel='freqency (Hz)', ylabel='psd', title='detrended psd')

            # remove negative values
            psd = self._remove_negative(psd)
            self.conditional_plot(freq_vals[i], psd.mean(axis=1), ax[8], plot_semiresults, xlabel='freqency (Hz)', ylabel='psd', title='nonnegative psd')

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
    def conditional_plot(x, y, ax=None, cond=False, xlabel='x', ylabel='y', title=''):
        """ Plots 'x' and 'y' to 'ax' as a line plot if 'cond' == True """
        if cond:
            ax.plot(x, y)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)

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
    def _autocorr(arr):
        """ calculate autocorrelation function from the array (arr) to reduce noise """
        N = arr.shape[0]  # počet vzorků signálu
        nmeas = arr.shape[1]  # počet měření

        Rrr = np.zeros((N-1, nmeas))

        for i in range(nmeas):
            a = arr[:, i]
            XX = hankel(a[1:])  # vytvoření hankelovy matice z prvků a[1] až a[N-1] (horní levá trojúhelníková matice)
            vX = a[:-1]  # vektor a[0] až a[N-2]
            Rrr[:, i] = np.matmul(XX, vX) / N - a.mean() ** 2  # výpočet normalizované ACF
        return Rrr

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
        return sum(par[i]*x**i for i in range(len(par)))


if __name__ == '__main__':

    plot_semiresults = True
    nmeas = 5

    p = Preprocessor()

    freqs, psd = p.run(["./data/trening/poruseno/2months/06202019_Acc.mat"], return_as="ndarray",
                       plot_semiresults=plot_semiresults, nmeas=nmeas)
