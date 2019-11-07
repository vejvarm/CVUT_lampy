import numpy as np

from matplotlib import pyplot as plt
from matplotlib import animation as animation

from preprocessing import Preprocessor
from Methods import Method


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


# TODO: load saved excited and unexcited arrays from files if they exist
if __name__ == '__main__':
    root = "D:/!private/Lord/Git/CVUT_lampy"
    datasets = ["neporuseno", "neporuseno2", "poruseno"]
    period = "2months"  # week or 2months
    paths = [f"{root}/data/{d}/{period}" for d in datasets]

    preprocessor = Preprocessor()

    for ds, path in zip(datasets, paths):
        freqs, psd_array = preprocessor.run([path], return_as="ndarray")

        # reshape psd_array from (nfiles, naccs, nfft/2, nmeas) to (nfiles*nmeas, naccs, nfft/2)
        psd_array = psd_array.transpose(0, 3, 1, 2)
        nfiles, nmeas, naccs, nfft = psd_array.shape
        psd_array = psd_array.reshape(nfiles*nmeas, naccs, nfft)

#        print(psd_array.shape)  # (nfiles*nmeas, naccs, nfft/2)

        # for each acc calculate mean values of PSD and individual psd measurements
        PSD = psd_array.mean(axis=0)
        PSD_mean = PSD.mean(axis=-1)
        PSD_snr = signaltonoise(PSD, axis=-1)
        psd_array_mean = psd_array.mean(axis=-1)
        psd_array_std = psd_array.std(axis=-1)
        psd_array_snr = signaltonoise(psd_array, axis=-1)

#        print(snr)
        print(PSD_snr)
        print(psd_array_snr.max(axis=0))
        print(psd_array_snr.min(axis=0))
        print(psd_array_snr.mean(axis=0))
        print(psd_array_snr.std(axis=0))

#        print(PSD_mean.shape)  # (naccs, )
#        print(psd_array_mean.shape)  # (nfiles*nmeas, naccs)

        # compute indices of excited and unexcited measurements (psd_mean >= PSD_mean]
#        ei_exc = np.where(psd_array_mean >= PSD_mean)   # Tuple[x index array, y index array]
#        ei_unexc = np.where(psd_array_mean < PSD_mean)  # Tuple[x index array, y index array]

        # based on SNR
        ei_exc = np.where(psd_array_snr <= psd_array_snr.mean(axis=0))
        ei_unexc = np.where(psd_array_snr > psd_array_snr.mean(axis=0))

        # split psd_array to excited and unexcited
        psd_array_excited = np.empty((nfiles*nmeas, naccs, nfft), dtype=np.float32)
        psd_array_unexcited = np.empty((nfiles*nmeas, naccs, nfft), dtype=np.float32)

        psd_array_excited[:] = np.nan
        psd_array_unexcited[:] = np.nan

        psd_array_excited[ei_exc[0], ei_exc[1]] = psd_array[ei_exc[0], ei_exc[1]]
        psd_array_unexcited[ei_unexc[0], ei_unexc[1]] = psd_array[ei_unexc[0], ei_unexc[1]]

        # reshape back to (nfiles, naccs, nfft/2, nmeas)
        psd_array_excited = psd_array_excited.reshape((nfiles, nmeas, naccs, nfft)).transpose(0, 2, 3, 1)
        psd_array_unexcited = psd_array_unexcited.reshape((nfiles, nmeas, naccs, nfft)).transpose(0, 2, 3, 1)

        num_excited = np.count_nonzero(np.all(~np.isnan(psd_array_excited), axis=2))
        num_unexcited = np.count_nonzero(np.all(~np.isnan(psd_array_unexcited), axis=2))

        print(f"all: {nfiles*nmeas*naccs} \nexcited: {num_excited} \nunexcited: {num_unexcited}")

        # save excited and unexcited arrays for later use
        np.save(path+"/psd_array_excited.npy", psd_array_excited)
        np.save(path+"/psd_array_unexcited.npy", psd_array_unexcited)

        # make average of excited and unexcited measurements (ignoring the nan values)
        PSD_excited = np.nanmean(psd_array_excited, axis=(0, 3))
        PSD_unexcited = np.nanmean(psd_array_unexcited, axis=(0, 3))

        # plot excited and unexcited values
        fig_exc, axes_exc = plt.subplots(3, 2)
        fig_unexc, axes_unexc = plt.subplots(3, 2)
        axes_exc, axes_unexc = axes_exc.flatten(), axes_unexc.flatten()
        for i in range(naccs):
            # excited
            axes_exc[i].stem(freqs, PSD_excited[i, :], use_line_collection=True, markerfmt=' ')
            axes_exc[i].set_yscale("log")
            axes_exc[i].set_xlabel("f (Hz)")
            axes_exc[i].set_ylabel("PSD (1)")
            axes_exc[i].set_ylim(None, 10)
            # unexcited
            axes_unexc[i].stem(freqs, PSD_unexcited[i, :], use_line_collection=True, markerfmt=' ')
            axes_unexc[i].set_yscale("log")
            axes_unexc[i].set_xlabel("f (Hz)")
            axes_unexc[i].set_ylabel("PSD (1)")
            axes_unexc[i].set_ylim(None, 10)

        fig_exc.suptitle(f"Průměr vybuzených ({ds})")
        fig_unexc.suptitle(f"Průměr nevybuzených ({ds})")

    plt.show()