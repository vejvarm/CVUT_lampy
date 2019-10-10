import numpy as np

from scipy import signal


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
            area = np.arange(peaks[j]-delta if peaks[j]-delta >= 0 else 0,
                             peaks[j]+delta if peaks[j]+delta < n_fft else n_fft - 1)
            idxs = area
            vals = psd[area]
            peaks_com_arr[i, j] = np.dot(idxs, vals)/vals.sum()

    return peaks_com_arr/ns_per_hz
