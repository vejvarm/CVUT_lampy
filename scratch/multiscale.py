import numpy as np
from matplotlib import pyplot as plt

from preprocessing import Preprocessor


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

    # Preprocessor config
    ns_per_hz = 10
    freq_range = (0, 256)
    noise_f_rem = (2, 50, 100, 150, 200)
    noise_df_rem = (2, 5, 2, 5, 2)
    mov_filt_size = 10

    preproc = Preprocessor(ns_per_hz=ns_per_hz,
                           freq_range=freq_range,
                           noise_f_rem=noise_f_rem,
                           noise_df_rem=noise_df_rem,
                           mov_filt_size=mov_filt_size)

#    filename = "08072018_AccM"
#    path = ['../data/neporuseno/2months/' + filename + ".mat"]

    path = ["../data/neporuseno2/2months/"]

    preprocessed = preproc.run(path)

    for key, (freqs, psd_list, _, _) in preprocessed.items():
        psd_array = np.array(psd_list)
        if 'psd_stacked' in locals():
            psd_stacked = np.concatenate((np.expand_dims(psd_array, 0), psd_stacked))
        else:
            psd_stacked = np.expand_dims(psd_array, 0)

    # average through the days
    psd_average = np.mean(psd_stacked, axis=0)

    naccs, nfreqs, nmeas = psd_average.shape
    bin_size = 26

    psd_bins = mean_to_bins(psd_average, bin_size)

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

    fig.savefig('../results/fig.png', dpi=300)
    fig2.savefig('../results/fig2.png', dpi=300)

    plt.show()
