import numpy as np
from matplotlib import pyplot as plt

def mean_to_bins(psd_array, bin_size):
    """Vezme pole psd_array a rozdělí ho na košíky (podle frekvence).
    hodnoty košíku se zprůměrují do jedné hodnoty

    :param psd_array: vstupní pole psd hodnot (počet hodnot na frekvenční ose, počet měřeni)
    :param bin_size: požadovaná velikost košíku (počet hodnot NE frekvence)

    """

    nfreqs, nmeas = psd_array.shape

    nbins = nfreqs//bin_size + (nfreqs%bin_size > 0)

    psd_bins = np.zeros((nbins, nmeas))

    for i in range(nbins):
        psd_bins[i, :] = psd_array[i*bin_size:(i+1)*bin_size, :].mean(axis=0)

    return psd_bins


if __name__ == "__main__":
    nfreqs = 2561
    nmeas = 144
    psd_array = np.arange(nfreqs*nmeas).reshape((nfreqs, nmeas))

    bin_size = 256

    psd_bins = mean_to_bins(psd_array, bin_size)

    fig, ax = plt.subplots(2, 1)
    im0 = ax[0].pcolormesh(psd_array)
    im1 = ax[1].pcolormesh(psd_bins)
    fig.colorbar(im1, ax=ax)

    plt.figure()
    plt.plot(psd_array[0:2, :])

    plt.show()
