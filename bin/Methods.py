# Method 1 (Finding centres of mass in PSD and then comparing them)
import os

import numpy as np

from matplotlib import pyplot as plt
from scipy import signal

from bin.flags import FLAGS
from bin.helpers import console_logger
from bin.Preprocessor import Preprocessor

LOGGER = console_logger(__name__, "WARNING")


class Method:

    def __init__(self, preprocessor=Preprocessor(), from_existing_file=True, nmeas=144, from_preprocessed=False, lamp="l1"):
        """
        :param preprocessor: (obj) Preprocessor class instance which determines how to preprocess PSD files
        :param from_existing_file: (bool) whether to try to load from existing files or ignore them
        :param nmeas: (int) number of measurements in one day
        :param from_preprocessed: (bool) whether we are loading already preprocessed files or not
        :param lamp: (str) which lamp is to be processed?
        """
        self.preprocessor = preprocessor
        self.from_existing_file = from_existing_file
        self.nmeas = nmeas
        self.from_preprocessed = from_preprocessed
        self.lamp = lamp
        if "l1" in self.lamp.lower():
            self.acc_slice = slice(0, 2)
        elif "l2" in self.lamp.lower():
            self.acc_slice = slice(2, 4)
        elif "l3" in self.lamp.lower():
            self.acc_slice = slice(4, None)
        else:
            raise ValueError("lamp has to be one of 'l1', 'l2' or 'l3'")

    def _calc_mean_and_var(self, psd, period=None):
        naccs, nfft, ndm = psd.shape
        if period:
            nsamples = ndm//self.nmeas//period
        else:
            nsamples = 1

        mean = np.empty((nsamples, naccs, nfft))
        var = np.empty((nsamples, naccs, nfft))

        psd = np.array_split(psd, nsamples, axis=-1)  # split to cca "period" long entries
#        print(f"period: {period}, nsamples: {nsamples}, len(psd): {len(psd)}")
        # calculate mean and var from each entry
        for i in range(nsamples):
            mean[i, :, :] = psd[i].mean(axis=2)
            var[i, :, :] = psd[i].var(axis=2)

        return mean, var

    def _get_PSD(self, path, period=None, remove_0th_dim=False):
        """ load freqs and PSD if the preprocessed files already exist, otherwise calculate freqs and PSD and save them

        :param path: (string) path to files with data
        :param period: (int) number of days from which to aggregate
        :param remove_0th_dim: (bool) if true, removes first dimension from resulting PSD arrays

        :return (freqs(1Darray), mean(1Darray), var(1Darray)):
        """
        if ".npy" in os.path.splitext(path)[-1]:
            folder_path = os.path.split(path)[0]
        else:
            folder_path = path
        path_to_freqs = folder_path + "/freqs.npy"
        path_to_PSD = folder_path + "/PSD.npy"
        path_to_vars = folder_path + "/PSDvar.npy"

        try:
            if self.from_preprocessed:
                pth_gen = os.walk(path)
                for root, subs, files in pth_gen:
                    freqs = np.load(os.path.join(root, files.pop(files.index("freqs.npy"))))  # load frequency file
                    psd = [np.load(os.path.join(root, file)) for file in files if ".npy" in os.path.splitext(file)[-1]]

                    psd = np.dstack(psd)

                    # choose only specific lamp
                    psd = psd[self.acc_slice, :, :]

                    mean, var = self._calc_mean_and_var(psd, period)
                    break
            elif self.from_existing_file and "X" in path and ".npy" in path:
                nlamps = FLAGS.nlamps
                freqs, X = self.preprocessor.run([path], return_as="ndarray")

                X = np.split(X, nlamps, axis=0)  # reshape from (ndays.nmeas.nlamps, nfft//2, naccs//nlamps)
                X = np.dstack(X)                 # to (ndays.nmeas, nfft//2, naccs)
                psd = X.transpose((2, 1, 0))     # and then to (naccs, nfft//2, ndays.nmeas)

                mean, var = self._calc_mean_and_var(psd, period)

            elif self.from_existing_file:
                freqs = np.load(path_to_freqs)
                mean = np.load(path_to_PSD)
                var = np.load(path_to_vars)
            else:
                print("\nIgnoring existing files!")
                raise FileNotFoundError
        except FileNotFoundError:
            freqs, psd = self.preprocessor.run([path], return_as="ndarray")
            ndays, naccs, nfft, nmeas = psd.shape
            # reshape from (ndays, naccs, nfft/2, nmeas) to (naccs, nfft//2, ndays.nmeas)
            psd = psd.transpose((1, 2, 0, 3))
            psd = psd.reshape((naccs, nfft, ndays*nmeas))

            # calculate PSD (== long term average values of psd)
            mean, var = self._calc_mean_and_var(psd, period=None)

            # save freqs and PSD files
            if ".mat" not in path:
                np.save(path_to_freqs, freqs)
                np.save(path_to_PSD, mean)
                np.save(path_to_vars, var)

        if not period and remove_0th_dim:
            mean = mean.reshape(mean.shape[1:])
            var = var.reshape(var.shape[1:])

        return freqs, mean, var


class M1(Method):

    def __init__(self, preprocessor=Preprocessor(), delta_f=5, peak_distance=10, n_peaks=10,
                 from_existing_file=True, nmeas=144, var_scaled_PSD=False):
        """

        :param preprocessor: (obj) Preprocessor class instance which determines how to preprocess PSD files
        :param delta_f: (float or int) Range of frequencies around the peak from which to calculate the centre of mass
        :param peak_distance: (float or int) Minimum allowed distance between top peaks (frequency)
        :param n_peaks: (int) Number of peaks to pick as the top peaks
        :param var_scaled_PSD: (bool) scale the PSD by the variance of psd (PSD*PSDvar)?
        """
        super(M1, self).__init__(preprocessor, from_existing_file=from_existing_file, nmeas=nmeas)

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

        freqs, PSD, PSD_var = self._get_PSD(path, remove_0th_dim=True)

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

        freqs, PSD, PSD_var = self._get_PSD(path, remove_0th_dim=True)

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

    def __init__(self, preprocessor=Preprocessor(), from_existing_file=True, nmeas=144, var_scaled_PSD=False, from_preprocessed=False, lamp="l1"):
        super(M2, self).__init__(preprocessor, from_existing_file=from_existing_file, nmeas=nmeas, from_preprocessed=from_preprocessed, lamp=lamp)

        self.bin_sizes = None
        self.thresholds = None
        self.var_scaled_PSD = var_scaled_PSD

        self.trained_distributions = None

    def train(self, path, bin_sizes, thresholds):
        """ calculate binarized distributions of PSD from path for each combination of bin_size and threshold nd then save them for further use

        :param path: path to file(s) containing psd arrays for training
        :param bin_sizes: tuple of desired bin sizes
        :param thresholds: tuple of desired thresholds
        :return: None
        """

        self.bin_sizes = bin_sizes
        self.thresholds = thresholds

        self.trained_distributions = self.get_multiscale_distributions(path, self.bin_sizes, self.thresholds)
        LOGGER.debug(f"trained_distributions.shape: {[td[2].shape for td in self.trained_distributions]}")

        print("Training complete!")

    def compare(self, path, period=None, print_results=True):
        """ Calculate cross entropy between psd loaded from path and trained multiscale distributions (M2().train).
        If period is specified, the compared psd files will first be split by days (e.g. period=1 means that ce will be
        calculated for each day of psd_comp sepparately)

        :param path: path to file(s) containing psd arrays for comparison with training data
        :param period: (int) split the comparison into periods of days (None==period is same as length of the data)
        :param print_results: (int)
        :return ce: array of cross entropies for each threshold, bin_size and period
        """

        if not self.trained_distributions:
            raise(ValueError, "Nejdříve je třeba metodu natrénovat (M2().train).")

        valid_distributions = self.get_multiscale_distributions(path, self.bin_sizes, self.thresholds, period)

        nperiods = len(valid_distributions)
        nbins = len(self.bin_sizes)
        nth = len(self.thresholds)

        ce = np.empty((nperiods, nbins*nth))

        if print_results:
            print("\n--CROSS ENTROPY VALUES--")
            print(f"|| period | bs |  th  || CEnt ||")
        for per, period_dist in enumerate(valid_distributions):
            for i, (((bin_sz, th), _, dtrain), ((_, _), _, dval)) in enumerate(zip(self.trained_distributions, period_dist)):
                ce[per, i] = self._cross_entropy(dtrain, dval)
                if print_results:
                    print(f'|| {per:6d} | {bin_sz:2d} | {th:.2f} || {ce[per, i]:4.0f} ||')

        return ce

    def get_multiscale_distributions(self, path, bin_sizes=(10, ), thresholds=(0.1, ), period=None):
        """ Calculate multiscale distributions from files given in path for each combination of values in
        bin_sizes and thresholds.

        :param path: path to files with psd files
        :param bin_sizes: tuple with desired bin sizes
        :param thresholds: tuple with desired thresholds
        :param period: period with which to calculate the distributions
        :return multiscale_distributions: List[(1, 1), 1D array[nbins], 2D array[naccs, nbins]]
        """

        freqs, PSD, PSD_var = self._get_PSD(path, period)
        LOGGER.debug(f"PSD.shape: {PSD.shape}")

        multiscale_distributions = list()

        for i, (mean, var) in enumerate(zip(PSD, PSD_var)):
            if self.var_scaled_PSD:
                var = (var - var.min())/(var.max() - var.min())  # normalize to interval (0, 1)
                mean = mean*var

            md = list()

            # multiscale (grid search way)
            for bin_size in bin_sizes:
                for threshold in thresholds:
                    freq_bins, PSD_bins = self._split_to_bins(freqs, mean, bin_size)

                    freq_binarized_mean = freq_bins.mean(axis=-1)
                    PSD_binarized_softmax = self._binarize_and_softmax(PSD_bins, threshold)

                    md.append([(bin_size, threshold), freq_binarized_mean, PSD_binarized_softmax])

            multiscale_distributions.append(md)

        return multiscale_distributions if period else multiscale_distributions[0]

    @staticmethod
    def _cross_entropy(d1, d2):
        """ Calculate information cross-entropy between d1 and d2

        :param d1: distribution 1
        :param d2: distribution 2
        :return: -sum(d1(x).log2(d2(x)))
        """

        return -np.sum(d1*np.log2(d2))


    @staticmethod
    def _split_to_bins(freqs, psd_array, bin_size):
        """Take psd_array and split it into bins (frequency-wise).

        :param freqs: array of input frequencies [number of frequencies, ]
        :param psd_array: array of input psd values [number of measurements, number of frequencies]
        :param bin_size: (int) desired bin size in number of values (not frequencies)

        :return freq_bins (2D array) [number of bins, size of one bin]
                psd_bins: (3D array) [number of measurements, number of bins, size of one bin]
        """
        naccs, nfreqs = psd_array.shape

        nbins = nfreqs // bin_size + (nfreqs % bin_size > 0)

        freq_bins = np.zeros((nbins, bin_size), dtype=np.float32)
        psd_bins = np.zeros((naccs, nbins, bin_size), dtype=np.float32)

        for i in range(nbins):
            area = slice(i*bin_size, (i + 1)*bin_size)
            fbin = freqs[area]
            pbin = psd_array[:, area]
            LOGGER.debug(f"fbin.shape: {fbin.shape}")
            if fbin.shape[0] < bin_size:
                LOGGER.info(f"Padding last bin to match shape!")
                fbin = np.pad(fbin, (0, bin_size - fbin.shape[0]), constant_values=np.max(fbin))
                LOGGER.debug(f"fbin: {fbin}")
                pbin = np.pad(pbin, ((0, 0), (0, bin_size - pbin.shape[1])), constant_values=0)
            freq_bins[i, :] = fbin
            psd_bins[:, i, :] = pbin
        LOGGER.debug(f"psd_bins.shape: {psd_bins.shape}")

        return freq_bins, psd_bins

    @staticmethod
    def _binarize_and_softmax(psd_bins, threshold):
        """ Calculate binarized values in bins scaled by softmax to probability distribution with sum of 1

        :param psd_bins: (3D array) psd array which is split to bins [:, nbins, bin_size]
        :param threshold: (int) desired cutoff value for binarization
        :return psd_binarized_softmaxed: (2D array) [:, nbins]
        """
        psd_binarized = np.array(psd_bins > threshold, dtype=np.float32)

        psd_binarized_sum = psd_binarized.sum(axis=-1)
        psd_binarized_sum = (psd_binarized_sum - psd_binarized_sum.mean())/psd_binarized_sum.std()
        psd_sum_of_exp = np.exp(psd_binarized_sum).sum(axis=-1)

        psd_binarized_softmaxed = np.exp(psd_binarized_sum)/np.expand_dims(psd_sum_of_exp, 1)

        return psd_binarized_softmaxed


if __name__ == '__main__':
    setting = "training"
    folder = FLAGS.paths[setting]["folder"]
    dataset = FLAGS.paths[setting]["dataset"]
    period = [FLAGS.paths[setting]["period"]]*len(dataset)
    filename = ["X.npy"]*len(dataset)
    paths = [f"./{folder}/{d}/{p}/{f}" for d, p, f in zip(dataset, period, filename)]

    print(paths)

    preprocessor = Preprocessor()

    from_existing_file = True
    var_scaled_PSD = False

    # METHOD 1 ---------------------------------------------------------------------------------------------------------
    # find peaks and learn centres of mass of neporuseno
    delta_f = 5
    peak_distance = delta_f * 2
    n_peaks = 3

    # Create M1 instance
    m1 = M1(preprocessor, delta_f=delta_f, peak_distance=peak_distance, n_peaks=n_peaks,
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
    m2 = M2(preprocessor, var_scaled_PSD=var_scaled_PSD)

    # multiscale params
    bin_sizes = (5, )
    thresholds = (0.1, 0.2)
    plot_distributions = False

    # Train the method on 2 months of neporuseno
    m2.train(paths[0], bin_sizes, thresholds)

    ce2 = m2.compare(paths[1], period=1, print_results=False)
    ce3 = m2.compare(paths[2], period=1, print_results=False)

#    print(ce2.shape)  # (nperiods, nbins*nthresholds)

    ce_diff = ce3 - ce2

    nbins = len(bin_sizes)
    nth = len(thresholds)

    for i, day in enumerate(ce_diff):
        daily_best = 0
        for j, val in enumerate(day):
            if val > daily_best:
                daily_best = val
                best_j = j
        print(f"day: {i}, bs: {bin_sizes[best_j%nbins]}, th: {thresholds[best_j%nth]} val: {daily_best}")

    distributions_1 = m2.get_multiscale_distributions(paths[0], bin_sizes=bin_sizes, thresholds=thresholds)
    distributions_2 = m2.get_multiscale_distributions(paths[1], bin_sizes=bin_sizes, thresholds=thresholds)
    distributions_3 = m2.get_multiscale_distributions(paths[2], bin_sizes=bin_sizes, thresholds=thresholds)

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
                nrows = 1
                ncols = 1
                fig, axes = plt.subplots(nrows, ncols)
                if nrows+ncols == 2:
                    axes = np.array([axes])
                for i, ax in enumerate(axes.flatten()):
                    y_pos = np.arange(len(d[i, :]))
                    ax.bar(y_pos, d[i, :], align="center", width=0.9)
#                    ax.set_xticks(np.arange(0, freqs.max(), 50))
                    ax.set_xlabel("frekvence (Hz)")
                    ax.set_ylabel("Softmax(psd_binarized) (1)")
                    ax.set_yscale("log")
                fig.suptitle(f"Binarizované spektrum {dataset[j]} | bs: {params[0]} | th: {params[1]} |")
    plt.show()

    # ------------------------------------------------------------------------------------------------------------------
