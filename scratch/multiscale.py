import numpy as np
from matplotlib import pyplot as plt

from bin.Preprocessor import Preprocessor


def binarize(psd_array, threshhold):
    """ make binarized psd array from input psd_array

    :param psd_array: (ndarray)
    :param threshhold: (float)
    :return: psd_binarized
    """

    bins = np.array([0.0, threshhold])

    psd_binarized = np.array(psd_array)
    psd_binarized[psd_binarized < threshhold] = 0.0
    psd_binarized[psd_binarized >= threshhold] = 1.0

    return psd_binarized


def mean_to_bins(psd_array, bin_size):
    """Vezme pole psd_array a rozdělí ho na košíky (podle frekvence).
    hodnoty košíku se zprůměrují do jedné hodnoty

    :param psd_array: vstupní pole psd hodnot (počet hodnot na frekvenční ose, počet měřeni)
    :param bin_size: požadovaná velikost košíku (počet hodnot NE frekvence)

    """

    naccs, nfreqs, nmeas = psd_array.shape

    nbins = nfreqs//bin_size + (nfreqs%bin_size > 0)

    psd_bins = np.zeros((naccs, nbins, nmeas))

    for i in range(nbins):
        psd_bins[:, i, :] = psd_array[:, i*bin_size:(i+1)*bin_size, :].mean(axis=1)

    return psd_bins


if __name__ == "__main__":
#    nfreqs = 2561
#    nmeas = 144
#    psd_array = np.arange(nfreqs*nmeas).reshape((nfreqs, nmeas))

    path = ["../data/neporuseno/week/"]
#    filename = "08072018_AccM"
#    path = ['../data/neporuseno/2months/' + filename + ".mat"]

    # load path and preprocess with default config
    preprocessor = Preprocessor()
    freqs, psd_stacked = preprocessor.run(path, return_as='ndarray')

    ns_per_hz = preprocessor.get_config_values()[1]

    # average through the days
    psd_average = np.mean(psd_stacked, axis=0)

    naccs, nfreqs, nmeas = psd_average.shape
    bin_size = 26

    psd_bins = mean_to_bins(psd_average, bin_size)
    psd_binarized = binarize(psd_bins, psd_bins.mean())

    psd_binarized_aggregated = psd_binarized.sum(axis=2)

    print(psd_average.shape)
    print(psd_bins.shape)

    nbins = psd_bins.shape[1]

    fbins = np.arange(0, nbins)*bin_size/ns_per_hz
    msrmnts = np.arange(0, nmeas)

    print(fbins.shape, msrmnts.shape)

    fig, ax = plt.subplots(3, 2)
    for i, a in enumerate(ax.flatten()):
        im = a.pcolormesh(freqs, msrmnts, psd_average[i, :, :].T)
    fig.colorbar(im, ax=ax)

    fig2, ax2 = plt.subplots(3, 2)
    for i, a in enumerate(ax2.flatten()):
        im = a.pcolormesh(fbins, msrmnts, psd_bins[i, :, :].T)
    fig2.colorbar(im, ax=ax2)

    fig3, ax3 = plt.subplots(3, 2)
    for i, a in enumerate(ax3.flatten()):
        im = a.pcolormesh(fbins, msrmnts, psd_binarized[i, :, :].T)
    fig3.colorbar(im, ax=ax3)

    fig4, ax4 = plt.subplots(3, 2)
    for i, a in enumerate(ax4.flatten()):
        plot = a.stem(fbins, psd_binarized_aggregated[i, :], use_line_collection=True, markerfmt=" ")

    fig.savefig('../results/original.png', dpi=300)
    fig2.savefig('../results/binned.png', dpi=300)
    fig3.savefig('../results/binarized.png', dpi=300)
    fig4.savefig('../results/binarized_aggregated.png', dpi=300)

    plt.show()
