import numpy as np

from scipy import signal
from matplotlib import pyplot as plt

from preprocessing import Preprocessor


def find_top_peaks(psd, delta_f, n_freqs):

    peaks, _ = signal.find_peaks(psd, distance=delta_f)
    sorted_peaks = sorted(peaks, key=lambda idx: psd[idx], reverse=True)

    if len(sorted_peaks) > n_freqs:
        return sorted_peaks[:n_freqs]
    else:
        return sorted_peaks


if __name__ == '__main__':
    ns_per_hz = 10
    delta_f = ns_per_hz*5//2
    n_freqs = 10

    path = ["../data/PovlakPrycSloup1OdleptaniSloup2Sloup3/2months"]

    preprocessor = Preprocessor()
    freqs, psd_stacked = preprocessor.run(path, return_as='ndarray')

    print(psd_stacked.shape)

    psd_average = psd_stacked.mean(axis=(0, 3))

    psd_average_0 = psd_average[0, :]

    # Finding (learning) peaks
    top_sorted_peaks = np.zeros((len(psd_average), n_freqs), dtype=np.int32)
    for i, psd in enumerate(psd_average):
        top_sorted_peaks[i, :] = find_top_peaks(psd, delta_f, n_freqs)

    # TODO: Finding centre of mass (těžiště)

    # TODO: Comparing centres of mass of the learned peak frequencies

    # Plotting
    fig, axes = plt.subplots(3, 2)
    for i, ax in enumerate(axes.flatten()):
        ax.plot(freqs, psd_average[i, :])
        ax.stem(freqs[top_sorted_peaks[i, :]], psd_average[i, top_sorted_peaks[i, :]], '-rx', use_line_collection=True)
    plt.show()
