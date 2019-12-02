# Method 1 (Finding centres of mass in PSD and then comparing them)
import os

import numpy as np

from os import path
from matplotlib import pyplot as plt
from scipy import signal

from preprocessing import Preprocessor


class Method:

    def __init__(self, preprocessor=Preprocessor(), from_existing_file=True):
        """
        :param preprocessor: (obj) Preprocessor class instance which determines how to preprocess PSD files
        :param from_existing_file: (bool) whether to try to load from existing files or ignore them
        """
        self.preprocessor = preprocessor
        self.from_existing_file = from_existing_file

    def _get_PSD(self, path):
        """ load freqs and PSD if the preprocessed files already exist, otherwise calculate freqs and PSD and save them

        :param path: (string) path to files with data
        """
        if ".npy" in os.path.splitext(path)[-1]:
            folder_path = os.path.split(path)[0]
        else:
            folder_path = path
        path_to_freqs = folder_path + "/freqs.npy"
        path_to_PSD = folder_path + "/PSD.npy"
        path_to_vars = folder_path + "/PSDvar.npy"

        try:
            if self.from_existing_file and "X.npy" in path:
                nlamps = 3
                freqs, X = self.preprocessor.run([path], return_as="ndarray")

                X = np.split(X, nlamps, axis=0)  # reshape from (ndays.nmeas.nlamps, nfft//2, naccs//nlamps)
                X = np.dstack(X)                 # to (ndays.nmeas, nfft//2, naccs)
                psd = X.transpose((2, 1, 0))     # and then to (naccs, nfft//2, ndays.nmeas)

                mean = psd.mean(axis=2)
                var = psd.var(axis=2)
            elif self.from_existing_file:
                freqs = np.load(path_to_freqs)
                mean = np.load(path_to_PSD)
                var = np.load(path_to_vars)
            else:
                print("\nIgnoring existing files!")
                raise FileNotFoundError
        except FileNotFoundError:
            freqs, psd = self.preprocessor.run([path], return_as="ndarray")

            # calculate PSD (== long term average values of psd)
            mean = np.nanmean(psd, axis=(0, 3))
            var = np.nanvar(psd, axis=(0, 3))

            # save freqs and PSD files
            np.save(path_to_freqs, freqs)
            np.save(path_to_PSD, mean)
            np.save(path_to_vars, var)

        return freqs, mean, var


class M1(Method):

    def __init__(self, preprocessor=Preprocessor(), delta_f=5, peak_distance=10, n_peaks=10,
                 from_existing_file=True, var_scaled_PSD=False):
        """

        :param preprocessor: (obj) Preprocessor class instance which determines how to preprocess PSD files
        :param delta_f: (float or int) Range of frequencies around the peak from which to calculate the centre of mass
        :param peak_distance: (float or int) Minimum allowed distance between top peaks (frequency)
        :param n_peaks: (int) Number of peaks to pick as the top peaks
        :param var_scaled_PSD: (bool) scale the PSD by the variance of psd (PSD*PSDvar)?
        """
        super(M1, self).__init__(preprocessor, from_existing_file=from_existing_file)

        self.preprocessor = preprocessor
        self.ns_per_hz = preprocessor.ns_per_hz
        self.delta_samples = int(self.ns_per_hz * delta_f)
        self.peak_distance = int(self.ns_per_hz * peak_distance)
        self.n_peaks = n_peaks
        self.var_scaled_PSD = var_scaled_PSD

        self.peaks = None
        self.centres_of_mass = None

    def train(self, path):
        """ learn top peak indices and their centres of mass"""

        freqs, PSD, PSD_var = self._get_PSD(path)

        if self.var_scaled_PSD:
            PSD_var = (PSD_var - PSD_var.min()) / (PSD_var.max() - PSD_var.min())  # normalize to interval (0, 1)
            PSD = PSD * PSD_var

        # calculate peak and centre of mass positions
        peaks = self.find_top_peaks(PSD, self.peak_distance, self.n_peaks)
        centres_of_mass = self.calc_centre_of_mass(PSD, peaks, self.delta_samples, self.ns_per_hz)

        # save as instance variables
        self.peaks = peaks
        self.centres_of_mass = centres_of_mass

        print("Training complete")

    def compare(self, path):
        """ calculate centres of mass (COM) at the learned peak indices and compare them with the trained COM

        :param path: path to folder with files that should be aggregated and compared with trained COM

        :return: sum of squares of differences between trained COM and COM calculated from path
        """

        freqs, PSD, PSD_var = self._get_PSD(path)

        if self.var_scaled_PSD:
            PSD_var = (PSD_var - PSD_var.min()) / (PSD_var.max() - PSD_var.min())  # normalize to interval (0, 1)
            PSD = PSD * PSD_var

        centres_of_mass = self.calc_centre_of_mass(PSD, self.peaks, self.delta_samples, self.ns_per_hz)

        return np.square(self.centres_of_mass - centres_of_mass).sum()

    @staticmethod
    def find_top_peaks(psd, peak_distance, n_peaks):
        """

        :param psd: (2D array) [n_entries, nfft//2]
        :param peak_distance: (int) [1] minimum peak distance in the resulting set of peaks
        :param n_peaks: (int) [1] maximum number of found peaks in psd
        :return top_sorted_peaks: (2D array) [n_entries, n_peaks]
        """
        n_entries = psd.shape[0]
        top_sorted_peaks = np.zeros((n_entries, n_peaks), dtype=np.int32)

        for i in range(n_entries):
            peaks, _ = signal.find_peaks(psd[i, :], distance=peak_distance)
            sorted_peaks = sorted(peaks, key=lambda idx: psd[i, idx], reverse=True)

            n_found_peaks = len(sorted_peaks)

            if n_found_peaks > n_peaks:
                top_sorted_peaks[i, :] = sorted_peaks[:n_peaks]
            else:
                top_sorted_peaks[i, :n_found_peaks] = sorted_peaks

        return top_sorted_peaks

    @staticmethod
    def calc_centre_of_mass(psd_arr, peaks_arr, delta, ns_per_hz):
        """

        :param psd_arr: (2D array) [:, nfft//2] power spectral density array
        :param peaks_arr: (2D array) [:, n_peaks] array with peak indices in psd
        :param delta: (int) [1] the area around peaks from which the centre of mass is to be calculated
        :param ns_per_hz: (int) [1] sampling frequency of the psd array
        :return peak_com_arr: (2D array) [:, n_peaks] array with calculated centre of mass FREQUENCIES!
        """

        n_accs, n_peaks = peaks_arr.shape
        n_fft = psd_arr.shape[1]

        peaks_com_arr = np.zeros((n_accs, n_peaks), dtype=np.float32)

        for i, (psd, peaks) in enumerate(zip(psd_arr, peaks_arr)):
            for j in range(n_peaks):
                area = np.arange(peaks[j] - delta if peaks[j] - delta >= 0 else 0,
                                 peaks[j] + delta if peaks[j] + delta < n_fft else n_fft - 1)
                idxs = area
                vals = psd[area]
                peaks_com_arr[i, j] = np.dot(idxs, vals) / vals.sum()

        return peaks_com_arr / ns_per_hz


class M2(Method):

    # rozdělit na biny
    # najít, které frekvence jsou vybuzené a které ne (multiscale threshold)
    # spočítat počet vybuzených
    # výpočet křížové entropie

    def __init__(self, preprocessor=Preprocessor(), from_existing_file=True, var_scaled_PSD=False):
        super(M2, self).__init__(preprocessor, from_existing_file=from_existing_file)

        self.bin_sizes = None
        self.thresholds = None
        self.var_scaled_PSD = var_scaled_PSD

    def get_multiscale_distributions(self, path, bin_sizes=(10, ), thresholds=(0.1, )):
        """

        :return multiscale_distributions: List[(1, 1), 1D array[nbins] 2D array[naccs, nbins]]
        """

        freqs, PSD, PSD_var = self._get_PSD(path)

        if self.var_scaled_PSD:
            PSD_var = (PSD_var - PSD_var.min())/(PSD_var.max() - PSD_var.min())  # normalize to interval (0, 1)
            PSD = PSD*PSD_var


        multiscale_distributions = list()

        # multiscale (grid search way)
        for bin_size in bin_sizes:
            for threshold in thresholds:
                freq_bins, PSD_bins = self._split_to_bins(freqs, PSD, bin_size)

                freq_binarized_mean = freq_bins.mean(axis=-1)
                PSD_binarized_softmax = self._binarize_and_softmax(PSD_bins, threshold)

                multiscale_distributions.append([(bin_size, threshold), freq_binarized_mean, PSD_binarized_softmax])

        return multiscale_distributions

    @staticmethod
    def _cross_entropy(d1, d2):
        """

        :param d1: distribution 1
        :param d2: distribution 2
        :return: -sum(d1(x).log(d2(x)))
        """

        return -np.sum(d1*np.log(d2))


    @staticmethod
    def _split_to_bins(freqs, psd_array, bin_size):
        """Vezme pole psd_array a rozdělí ho na košíky (podle frekvence).

        :param freqs: vstupní pole hodnot frekvencí [počet hodnot na frekvenční ose, ]
        :param psd_array: vstupní pole psd hodnot (počet měření, počet hodnot na frekvenční ose)
        :param bin_size: (int) požadovaná velikost košíku (počet hodnot NE frekvence)

        :return freq_bins (2D array) [počet binů, velikost jednoho binu]
                psd_bins: (3D array) [počet měření, počet binů, počet frekvencí v jednom binu]
        """

        naccs, nfreqs = psd_array.shape

        nbins = nfreqs // bin_size + (nfreqs % bin_size > 0)

        freq_bins = np.zeros((nbins, bin_size), dtype=np.float32)
        psd_bins = np.zeros((naccs, nbins, bin_size), dtype=np.float32)

        for i in range(nbins):
            area = slice(i*bin_size, (i + 1)*bin_size)
            freq_bins[i, :] = freqs[area]
            psd_bins[:, i, :] = psd_array[:, area]

        return freq_bins, psd_bins

    @staticmethod
    def _binarize_and_softmax(psd_bins, threshold):
        """

        :param psd_bins: (3D array) psd array which is split to bins [:, nbins, bin_size]
        :param threshold: (int) desired cutoff value for binarization
        :return psd_binarized_softmaxed: (2D array) [:, nbins]
        """

        psd_binarized = np.array(psd_bins > threshold, dtype=np.float32)

        psd_binarized_sum = psd_binarized.sum(axis=-1)

        psd_binarized_softmaxed = np.exp(psd_binarized_sum)/np.expand_dims(np.exp(psd_binarized_sum).sum(axis=-1), 1)

        return psd_binarized_softmaxed


if __name__ == '__main__':
    folder = "trening"
    dataset = ["neporuseno", "neporuseno2", "poruseno"]
    period = "2months"  # week or 2months
    filename = "X.npy"
    paths = [f"./data/{folder}/{d}/{period}/{filename}" for d in dataset]

    preprocessor = Preprocessor()

    from_existing_file = True
    var_scaled_PSD = False

    # METHOD 1 ---------------------------------------------------------------------------------------------------------
    # find peaks and learn centres of mass of neporuseno
    delta_f = 5
    peak_distance = delta_f * 2
    n_peaks = 3

    # Create M1 instance
    m1 = M1(Preprocessor(), delta_f=delta_f, peak_distance=peak_distance, n_peaks=n_peaks,
            from_existing_file=from_existing_file, var_scaled_PSD=var_scaled_PSD)

    # Train the instance (learn top peaks and their COM)
    m1.train(paths[0])

    # Compare learned COM with COM from PSD at given paths
    sum_diff_nepor2 = m1.compare(paths[1])
    sum_diff_por = m1.compare(paths[2])

    print(f"____________METHOD 1___________")
    print(f"________##### {n_peaks} #####_________")
    print(f"Sum of square differences neporuseno2: {sum_diff_nepor2}")
    print(f"Sum of square differences poruseno: {sum_diff_por}")
    # ------------------------------------------------------------------------------------------------------------------

    # METHOD 2 ---------------------------------------------------------------------------------------------------------
    m2 = M2(var_scaled_PSD=var_scaled_PSD)

    # multiscale params
    bin_sizes = (40, )
    tresholds = (0.4, )
    plot_distributions = True

    distributions_1 = m2.get_multiscale_distributions(paths[0], bin_sizes=bin_sizes, thresholds=tresholds)
    distributions_2 = m2.get_multiscale_distributions(paths[1], bin_sizes=bin_sizes, thresholds=tresholds)
    distributions_3 = m2.get_multiscale_distributions(paths[2], bin_sizes=bin_sizes, thresholds=tresholds)

    print("\n--CROSS ENTROPY VALUES--")
    print(f"| bs | trsh || ds2 | ds3 | diff |")
    for ((bin_sz, th), _, d1), ((_, _), _, d2), ((_, _), _, d3) in zip(distributions_1, distributions_2, distributions_3):
        ce2 = M2._cross_entropy(d1, d2)
        ce3 = M2._cross_entropy(d1, d3)
        ce_diff = ce3 - ce2
        print(f'| {bin_sz:2d} | {th:.2f} || {ce2:3.0f} | {ce3:3.0f} | {ce_diff:4.0f} |')

    # plotting distributions
    if plot_distributions:
        for j, dist in enumerate((distributions_1, distributions_2, distributions_3)):
            for (params, freqs, d) in dist:
                fig, axes = plt.subplots(3, 2)
                for i, ax in enumerate(axes.flatten()):
                    ax.semilogy(freqs, d[i, :])
                fig.suptitle(f"{dataset[j]} | {params} |")
        plt.show()

    # ------------------------------------------------------------------------------------------------------------------
