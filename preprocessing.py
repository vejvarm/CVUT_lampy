import os
from copy import deepcopy

import numpy as np

from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from scipy.optimize import leastsq
from scipy.linalg import hankel
from tqdm import tqdm

from flags import FLAGS

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
                 tdf_order=5,
                 tdf_ranges=((45, 55), (95, 105), (145, 155), (195, 205)),
                 use_autocorr=True,
                 noise_f_rem=(0, ),
                 noise_df_rem=(0, ),
                 mov_filt_size=5,
                 rem_neg=True):
        """

        :param fs: (int) sampling frequency of the acquisition
        :param ns_per_hz: (int) desired number of samples per Hertz in FFT
        :param freq_range: (list/tuple) (minimum frequency, maximum frequency) rest is thrown away
        :param tdf_order: (int) order of the time domain bandreject filter
        :param tdf_ranges: (List/Tuple[List/Tuple]) time domain filter bandreject frequency areas ((lb1, ub1), (lb2, ub2), ...)
        :param use_autocorr: (bool) if True, calculate autocorrelation function before transforming to psd
        :param noise_f_rem: (list/tuple) frequencies that should be removed (zeroed out) from the power spectrum
        :param noise_df_rem: (list/tuple) range around f_rem that should also be removed
        :param mov_filt_size: (int) length of the rectangular filter for moving average application
        :param rem_neg: (bool) if True, remove negative values after the final preprocessing stage
        """
        self.fs = fs
        self.ns_per_hz = ns_per_hz
        self.freq_range = freq_range
        self.tdf_order = tdf_order
        self.tdf_ranges = np.array(tdf_ranges)  # 2D array
        self.use_autocorr = use_autocorr
        self.noise_f_rem = noise_f_rem
        self.noise_df_rem = noise_df_rem
        self.mov_filt_size = mov_filt_size
        self.rem_neg = rem_neg

        # calculate numerators and denominators of time domain frequency filters
        self.nums, self.denoms = self._make_bandstop_filters()

        # initialize counter for conditional plot
        self.cplot_call = 0

        # initialize dict for semiresults
        self.semiresults = dict()

    def get_config_values(self):
        return tuple(self.__dict__.values())

    def run(self, paths, return_as='dict', plot_semiresults=False, every=1, nmeas=None, savefolder=None):
        """

        :param paths: (list/tuple) strings of paths leading to .mat files or folder with .mat files for loading
        :param return_as: (string) if 'dict', returns a dictionary, if 'ndarray' returns an ndarray of only psd
        :param plot_semiresults: if True, make a plot after each preprocessing operation
        :param every: if > 1, only every 'every'eth acc will be plotted (e.g. every=2 means that every 2nd acc is plotted)
        :param nmeas: if None, take all measurements available in file, if (int), take first nmeas measurements
        :param savefolder: if None, don't save pdf of plot, else save pdf of plot to savefolder
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
                                freq_vals, psd_list, wind_dir, wind_spd = self._preprocess(fullpath, plot_semiresults, every, nmeas)
                                preprocessed[f_name] = (freq_vals, psd_list, wind_dir, wind_spd)
                            pbar.update(1)
            elif os.path.splitext(path)[-1] == '.mat':
                # leads to .mat file ... load directly
                freq_vals, psd_list, wind_dir, wind_spd = self._preprocess(path, plot_semiresults, every, nmeas)
                f_name = os.path.splitext(os.path.basename(path))[0]
                preprocessed[f_name] = (freq_vals, psd_list, wind_dir, wind_spd)
                # save plot to savefolder
                if plot_semiresults and savefolder:
                    os.makedirs(savefolder, exist_ok=True)
                    plt.savefig(os.path.join(savefolder, f"{f_name}.pdf"))
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

    @staticmethod
    def compare_arrs(x1, y1, x2, y2, title="Porovnání dvou vstupních polí.",
                              label1="y1", label2="y2", savefolder=None):
        fig, ax = plt.subplots(1, 1)

        ax.plot(x1, y1, label=label1)
        ax.plot(x2, y2, label=label2)

        ax.set_title(title)
        ax.set_xlabel("frekvence (Hz)")
        plt.legend()

        if savefolder:
            os.makedirs(savefolder, exist_ok=True)
            plt.savefig(os.path.join(savefolder, f"{label1}X{label2}.pdf"))

    def compare_arrs_with_psd(self, x1, y1, x2, y2, suptitle="Porovnání časových a frekvenčních oblastí.",
                              label1="y1", label2="y2", title1="y", title2="psd(y)", savefolder=None):
        fig, ax = plt.subplots(2, 2)

        ax[0, 0].plot(x1, y1, label=label1)
        ax[1, 0].plot(x2, y2, label=label2)
        ax[0, 1].plot(*self._calc_psd(np.expand_dims(y1, 1)))
        ax[1, 1].plot(*self._calc_psd(np.expand_dims(y2, 1)))

        plt.suptitle(suptitle)
        ax[0, 0].set_title(title1)
        ax[0, 1].set_title(title2)
        ax[1, 0].set_xlabel("čas (s)")
        ax[1, 1].set_xlabel("frekvence (Hz)")
        ax[0, 0].set_ylabel(label1)
        ax[0, 1].set_ylabel(f"psd({label1})")
        ax[1, 0].set_ylabel(label2)
        ax[1, 1].set_ylabel(f"psd({label2})")

        plt.subplots_adjust(top=0.886,
                            bottom=0.121,
                            left=0.122,
                            right=0.977,
                            hspace=0.223,
                            wspace=0.455)

        if savefolder:
            os.makedirs(savefolder, exist_ok=True)
            plt.savefig(os.path.join(savefolder, f"{label1}X{label2}.pdf"))

    def _preprocess(self, fullpath2file, plot_semiresults=False, every=1, nmeas=None):
        """ run the preprocessing pipeline

        :param fullpath2file: full path to .mat file that should be preprocessed
        :param plot_semiresults: if True, make a plot after each preprocessing operation
        :param every: if > 1, only every 'every'eth acc will be plotted (e.g. every=2 means that every 2nd acc is plotted)
        :param nmeas: if None, take all measurements available in file, if (int), take first nmeas measurements
        :return:
            :var freq_vals:  (1D ndarray) frequency values (x axis) [fs*ns_per_hz//2],
            :var psd_list: (List[2D ndarray]) preprocessed power spectral densities [self.nfft, number of measurements]
            :var wind_dir: (1D string array) [144] (dir. of wind described by characters N, S, E, W)
            :var wind_spd: (1D float array) [144] (speed of the wind)
        """
        df = self._mat2dict(fullpath2file)

        naccs = FLAGS.naccs
        freq_vals = [[]]*naccs
        psd_list = [[]]*naccs

        acc = [df[f'Acc{i}'] for i in range(1, naccs + 1)]

        if nmeas:
            for i in range(naccs):
                acc[i] = acc[i][:, :nmeas]

        # check if self.fs fits to value of 'FrekvenceSignalu'
        assert df['FrekvenceSignalu'] == self.fs, f"Value of 'FrekvenceSignalu' in {fullpath2file} doesn't correspond with self.fs ({self.fs} Hz)"

        # initialize plot if 'plot_semiresults' == True
        nops = 6 - int(not self.use_autocorr) - int(not self.rem_neg)  # number of preprocessing operations (for plotting)
        if plot_semiresults:
            ncols = naccs//every
            fig, ax = plt.subplots(nops, ncols)
            fig.set_size_inches([6.4, 12.8])
            if ncols <= 1:
                ax = np.expand_dims(ax, axis=-1)
            plt.subplots_adjust(top=0.958,
                                bottom=0.091,
                                left=0.103,
                                right=0.985,
                                hspace=1.0,
                                wspace=0.166)  # empirical setting on 27' FullHD monitor
        else:
            fig = None
            ax = np.empty((nops, naccs))
            ax[:] = np.nan

        for i, arr in enumerate(acc):

            # reset counter for conditional plot calls
            self.cplot_call = 0

            nvals, nmeas = arr.shape
            time = np.arange(nvals) / self.fs

            # normalize 'arr' to mean==0 and variance==1
            arr = self._calc_zscore(arr)
            self._conditional_plot(time, arr.mean(axis=1), plot_semiresults, ax, i, every,
                                   xlabel='čas (s)', ylabel='μ(arr)', key="01_norm",
                                   title=f'normalizovaný signál z akcelerometru')

            # remove area around 50 Hz noise using time domain filter
            arr = self._apply_time_domain_filters(arr)
            self._conditional_plot(time, arr.mean(axis=1), plot_semiresults, ax, i, every,
                                   xlabel='čas (s)', ylabel='μ(arr)-*50Hz', key="02_td_filt",
                                   title='filtrace násobků 50 Hz v čas. oblasti')

            # calculate autocorrelation function to reduce noise
            if self.use_autocorr:
                arr = self._autocorr(arr)
                self._conditional_plot(time[:-1], arr.mean(axis=1), plot_semiresults, ax, i, every,
                                       xlabel='čas (s)', ylabel='ρxx', key="03_autocorr",
                                       title='výpočet autokorelační funkce')

            # calculate frequency values and power spectral density (psd) of 'arr'
            freq_vals[i], psd = self._calc_psd(arr)

            # remove noise based on values in self.noise_f_rem and self.noise_df_rem
#            psd = self._remove_noise(psd, replace_with="min")
#            self._conditional_plot(freq_vals[i], psd.mean(axis=1), plot_semiresults, ax, i, every,
#                                  xlabel='frekvence(Hz)', ylabel='psd', title='freq domain filtering')

            # normalize 'psd' to mean==0 and variance==1
            psd = (psd - psd.mean())/psd.std()
            self._conditional_plot(freq_vals[i], psd.mean(axis=1), plot_semiresults, ax, i, every,
                                   xlabel='frekvence(Hz)', ylabel='psd', key="04_psd_norm",
                                   title='normalizovaná výkonová spektrální hustota (psd)')

            # use moving average to smooth out the spectrum and remove noise
            psd = self._coarse_grain(psd)
            self._conditional_plot(freq_vals[i], psd.mean(axis=1), plot_semiresults, ax, i, every,
                                   xlabel='frekvence(Hz)', ylabel='psd_norm', key="05_psd_cg",
                                   title=f'klouzavý průměr s krokem {self.mov_filt_size}')

            # remove trend
            psd = self._detrend(freq_vals[i], psd)

            # remove negative values
            if self.rem_neg:
                psd = self._remove_negative(psd)
                self._conditional_plot(freq_vals[i], psd.mean(axis=1), plot_semiresults, ax, i, every,
                                       xlabel='frekvence(Hz)', ylabel='psd_noneg', key="06_psd_noneg",
                                       title='odstranění trendu a negativních hodnot')

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

    def _conditional_plot(self, x, y, cond=False, ax=None, col=0, every=1, xlabel='x', ylabel='y', key='', title='', scale='linear'):
        """ Plots 'x' and 'y' to 'ax' as a line plot if 'cond' == True
            plot only when j % every == 0 (default 'every = 1' which means that every input is plotted if cond==True)
        """
        row = self.cplot_call
        jj = col // every
        if cond and ax[row, jj] and not col % every:
            ax[row, jj].plot(x, y)
            ax[row, jj].set_xlabel(xlabel)
            ax[row, jj].set_ylabel(ylabel)
            ax[row, jj].set_title(title)
            ax[row, jj].set_yscale(scale)
            if key in self.semiresults:
                self.semiresults[key].append((x, y))
            else:
                self.semiresults[key] = [(x, y)]

        self.cplot_call += 1

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
        """ calculate autocorrelation function (ACF) from the array (arr) to reduce noise
        """
        N = arr.shape[0]  # number of signal samples
        nmeas = arr.shape[1]  # number of measurements

        Rrr = np.zeros((N-1, nmeas))

        for i in range(nmeas):
            a = arr[:, i]
            a = np.fft.ifftshift((a - a.mean())/a.std())  # standardize and shift center to middle
            a = np.pad(a, (N//2, N//2), mode="constant")  # pad with zeros to simulate periodicity for FFT
            A = np.fft.fft(a)  # convert to frequency domain
            AA = np.abs(A)**2  # calculate autocorrelation in the frequency domain
            aa = np.fft.ifft(AA)[:N:-1]  # convert back to time domain and take only the relevant half of the signal
            Rrr[:, i] = np.real(aa)/N - a.mean()**2  # save normalized autocorrelation function value
        return Rrr

    def _make_bandstop_filters(self):
        """ calculate a bandstop filter params from given input parameters """
        f_nyquist = self.fs/2
        Wh = self.tdf_ranges/f_nyquist

        nfilts = Wh.shape[0]

        nums = np.empty((nfilts, 2*self.tdf_order + 1))
        denoms = np.empty((nfilts, 2*self.tdf_order + 1))

        for i in range(nfilts):
            b, a, *_ = butter(self.tdf_order, Wh[i, :], 'bandstop')
            nums[i, :] = b
            denoms[i, :] = a

        return nums, denoms

    def _apply_time_domain_filters(self, arr):
        """ apply all calculated bandstop filteres on arr """
        nfilts = self.nums.shape[0]

        for i in range(arr.shape[-1]):
            for num, denom in zip(self.nums, self.denoms):
                arr[:, i] = filtfilt(num, denom, arr[:, i])  # updating in place
        return arr

    @staticmethod
    def _hamming(arr):
        """ Apply Hamming window across the 0th dimension in arr """
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

    @staticmethod
    def __error_func(par, x, y, func):
        return np.square(y - func(par, x))

    @staticmethod
    def __linear_func(par, x):
        return sum(par[i]*x**i for i in range(len(par)))


if __name__ == '__main__':

    autocorr = True
    rem_neg = True
    plot_semiresults = True
    nmeas = 1
    every = 6
    acc_idx = 0
    plot_save_folder = f"./images/preprocessing/autocorr-{autocorr}/rem_neg-{rem_neg}/nmeas-{nmeas}/every-{every}/acc_idx-{acc_idx}/"

    p = Preprocessor(use_autocorr=autocorr, rem_neg=rem_neg)

    freqs, psd = p.run(["data/trening/neporuseno/2months/08072018_AccM.mat"], return_as="ndarray",
                       plot_semiresults=plot_semiresults, every=every, nmeas=nmeas, savefolder=plot_save_folder)


    time, norm_acc = p.semiresults["01_norm"][acc_idx]
    _, td_filt_acc = p.semiresults["02_td_filt"][acc_idx]
    time_corr, corr_acc = p.semiresults["03_autocorr"][acc_idx]
    freqs_, psd_norm = p.semiresults["04_psd_norm"][acc_idx]
    _, psd_cg = p.semiresults["05_psd_cg"][acc_idx]
    _, psd_noneg = p.semiresults["06_psd_noneg"][acc_idx]

    p.compare_arrs_with_psd(time, norm_acc, time, td_filt_acc,
                            suptitle="Odstranění násobků 50 Hz",
                            label1="norm_acc", label2="td_filt_acc",
                            title1="časová oblast", title2="frekvenční oblast",
                            savefolder=plot_save_folder)
    p.compare_arrs_with_psd(time, td_filt_acc, time_corr, corr_acc,
                            suptitle="Výpočet autokorelační funkce",
                            label1="td_filt_acc", label2="corr_acc",
                            title1="časová oblast", title2="frekvenční oblast",
                            savefolder=plot_save_folder)

    p.compare_arrs(freqs_, psd_norm, freqs_, psd_cg,
                   title=f"Klouzavý průměr s krokem {p.mov_filt_size}",
                   label1="psd_norm", label2="psd_cg",
                   savefolder=plot_save_folder)
    p.compare_arrs(freqs_, psd_cg, freqs_, psd_noneg,
                   title=f"Odstranění trendu a ořez hodnot",
                   label1="psd_cg", label2="psd_noneg",
                   savefolder=plot_save_folder)

    p.compare_arrs(freqs_, psd_noneg, 0, 0, title=f"Výstupní výkonová spektrální hustota singálu",
                   label1="psd_noneg", label2="",
                   savefolder=plot_save_folder)

    plt.show()
