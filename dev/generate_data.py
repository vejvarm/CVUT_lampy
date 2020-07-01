import os

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from bin.flags import FLAGS
from bin.helpers import console_logger, plotter
from bin.Generator import Generator
from bin.Preprocessor import Preprocessor

LOGGER = console_logger(__name__, "DEBUG")
PSNR_CSV_SETUP = FLAGS.PSNR_csv_setup


def generate_and_compute_snr(signal_gen, noise_gen, shape=(10, 15360)):
    assert len(shape) == 2, "given shape must be shape 2 Tuple or List"
    signal = signal_gen.generate(*shape)
    noise = noise_gen.generate(*shape)

#    s_power = np.sum(abs(signal)**2)/np.size(signal)
#    n_power = np.sum(abs(noise)**2)/np.size(noise)  # == MSE
#    LOGGER.debug(f"s_power: {s_power:.2f} | n_power: {n_power:.2f}")

#    s_max = np.max(signal)**2
#    n_max = np.max(noise)**2
#    LOGGER.debug(f"s_max: {s_max:.2f} | n_max: {n_max:.2f}")

#    SNR = s_power/n_power
#    PSNR = s_max/n_power
#    LOGGER.debug(f"SNR: {SNR:.3f} | PSNR: {PSNR:.3f}")

    SS = np.abs(np.fft.fft(signal))**2
    NN = np.abs(np.fft.fft(noise))**2
    PSNR_dB = 10*np.log(np.max(SS)/np.mean(NN))  # dB Signal to noise ratio

    LOGGER.debug(f"PSNR_dB: {PSNR_dB:.3f} dB")
    return signal + noise, PSNR_dB


def zero_out_freqs_pattern(x, fs, freqs=(50, 100, 200)):
    """

    :param x: (2D array) (num_sigs, samples) signal array
    :param fs: sampling frequency
    :param freqs: (tuple) tuple of frequencies to remove
    :return x: (2D array) without some frequencies
    """
    nsig, nsamp = x.shape
    nfreqs = len(freqs)
    x = np.fft.ifftshift(((x.T - x.mean(axis=-1))/x.std(axis=-1)).T, axes=-1)
    LOGGER.debug(f"x.mean: {x.mean()}")
    x = np.pad(x, ((0, 0), (nsamp//2, nsamp//2)), mode="constant")
    LOGGER.debug(f"x.mean: {x.mean()}")
    X = np.fft.fft(x, nsamp*2, axis=-1)
    f = np.arange(0., fs, fs/nsamp/2)
    LOGGER.debug(f"X.mean: {X.mean()}")
    LOGGER.debug(f"f.shape: {f.shape}")
    plt.figure()
    for i in range(nsig):
        ii = i % nfreqs
        idx = np.where(np.isclose(f, freqs[ii]))
        X[ii, idx] = 0
    plt.plot(X[0, :])
    return np.fft.ifft(X, nsamp*2, axis=-1).real[:, nsamp+nsamp//2:nsamp//2:-1]


if __name__ == '__main__':
    nrepeats = 1
    signal_amps = [(0.5, 1)]
    noise_amps = [(0, 0.5)]

    fs = 512
    nfft = 5120
    nsamples = 15360
    root = "./data/for_article/generated"
    plot_exts = [".pdf", ".svg"]

    dPSNR = {"episode": [], "signal_amp_max": [], "noise_amp_max": [], "train_PSNR (dB)": [], "test_PSNR (dB)": []}

    plot_divided = True

    for i in range(nrepeats):
        for s_amp in signal_amps:
            for n_amp in noise_amps:
                dtrain = {"nsig": 3, "fvs": (10., 80., 125.), "amp_range": s_amp,
                          "shift_range": (0, 0), "path": f"{root}/train/"}
                dtest = {"nsig": 6, "fvs": (10., 80., 125.), "amp_range": s_amp,
                         "shift_range": (0, 0), "path": f"{root}/test/"}
                dtest_br = {"nsig": 6, "fvs": (7., 83., 122.), "amp_range": s_amp,
                            "shift_range": (0, 0), "path": f"{root}/test/"}
                dnoise = {"fvs": np.arange(fs), "amp_range": n_amp, "shift_range": (0, 0)}

                LOGGER.info("Initialising signal and noise generators.")
                g_train = Generator(fs, fvs=dtrain["fvs"], amp_range=dtrain["amp_range"],
                                    shift_range=dtrain["shift_range"])
                g_test = Generator(fs, fvs=dtest["fvs"], amp_range=dtest["amp_range"], shift_range=dtest["shift_range"])
                g_test_br = Generator(fs, fvs=dtest_br["fvs"], amp_range=dtest_br["amp_range"],
                                      shift_range=dtest_br["shift_range"])
                g_noise = Generator(fs, fvs=dnoise["fvs"], amp_range=dnoise["amp_range"],
                                    shift_range=dnoise["shift_range"])

                LOGGER.info("Generating signal values with added noise.")
                t = np.arange(nsamples) / fs
                x_train, PSNR_train = generate_and_compute_snr(g_train, g_noise, shape=(dtrain["nsig"], nsamples))
                x_test_un, PSNR_test_un = generate_and_compute_snr(g_test, g_noise, shape=(dtest["nsig"], nsamples))
                x_test_br, PSNR_test_br = generate_and_compute_snr(g_test_br, g_noise,
                                                                   shape=(dtest_br["nsig"], nsamples))
                LOGGER.debug(f"x_train.shape: {x_train.shape}, "
                             f"x_test_un.shape: {x_test_un.shape}, "
                             f"x_test_br.shape: {x_test_br.shape}")

                LOGGER.info("Zeroing out certain natural frequencies.")
#                x_train = zero_out_freqs_pattern(x_train, fs, freqs=(dtrain["fvs"]))
#                x_test_un = zero_out_freqs_pattern(x_test_un, fs, freqs=(dtest["fvs"]))
#                x_test_br = zero_out_freqs_pattern(x_test_br, fs, freqs=(dtest_br["fvs"]))
                LOGGER.debug(f"x_train.shape: {x_train.shape}, "
                             f"x_test_un.shape: {x_test_un.shape}, "
                             f"x_test_br.shape: {x_test_br.shape}")


                LOGGER.info("Concatenating unbroken and broken test data")
                x_test = np.vstack([x_test_un, x_test_br])

                LOGGER.info("Preprocessing signals to psd using Preprocessor with default values.")
                p = Preprocessor()
                freq_train, psd_train = p.simple_preprocess(x_train.T)
                freq_test, psd_test = p.simple_preprocess(x_test.T)
                freq_test_un, psd_test_un = p.simple_preprocess(x_test_un.T)
                freq_test_br, psd_test_br = p.simple_preprocess(x_test_br.T)
                LOGGER.debug(f"psd_train.shape: {psd_train.shape}, psd_test.shape: {psd_test.shape}")
                LOGGER.debug(f"psd_test_un.shape: {psd_test_un.shape}, psd_test_br.shape: {psd_test_br.shape}")

                LOGGER.info("Reshaping to M2 compatible shape.")
                psd_train = np.expand_dims(psd_train.T, axis=-1)
                psd_test = np.expand_dims(psd_test.T, axis=-1)
                psd_test_un = np.expand_dims(psd_test_un.T, axis=-1)
                psd_test_br = np.expand_dims(psd_test_br.T, axis=-1)
                LOGGER.debug(f"psd_train.shape: {psd_train.shape}, psd_test.shape: {psd_test.shape}")

                LOGGER.debug("Making folder if it doesn't exist")
                os.makedirs(dtrain["path"], exist_ok=True)
                os.makedirs(dtest["path"], exist_ok=True)

                LOGGER.info("Generating save file names.")
                save_name_raw = f"raw{i}_sAmp{s_amp}_nAmp{n_amp}.npy"
                save_name_psd = f"X{i}_sAmp{s_amp}_nAmp{n_amp}.npy"
                freqs_name = f"freqs{i}.npy"

                LOGGER.info("Saving raw signals to files.")
                np.save(os.path.join(dtrain["path"], save_name_raw), x_train)
                LOGGER.debug(f"x_train saved to {dtrain['path']} with file name {save_name_raw}")
                np.save(os.path.join(dtest["path"], save_name_raw), x_test)
                LOGGER.debug(f"x_test saved to {dtest['path']} with file name {save_name_raw}")

                LOGGER.info("Saving generated arrays to files.")
                np.save(os.path.join(dtrain["path"], "freqs.npy"), freq_train)
                LOGGER.debug(f"freq_train saved to {dtrain['path']} with file name {freqs_name}")
                np.save(os.path.join(dtrain["path"], save_name_psd), psd_train)
                LOGGER.debug(f"psd_train saved to {dtrain['path']} with file name {save_name_psd}")
                np.save(os.path.join(dtest["path"], "freqs.npy"), freq_test)
                LOGGER.debug(f"freq_test saved to {dtest['path']} with file name {freqs_name}")
                np.save(os.path.join(dtest["path"], save_name_psd), psd_test)
                LOGGER.debug(f"psd_test saved to {dtest['path']} with file name {save_name_psd}")

                # Vykreslení výsledků
                LOGGER.info("Plotting results.")
                _, ax = plt.subplots(3, 1)
                LOGGER.info("Plotting noisy signals.")
                plt.suptitle("Noisy signals"
                             f"\n signal: {s_amp}, noise: {n_amp}")
                plotter(t, x_train.mean(0), ax[0], title="", ylabel="train")
                plotter(t, x_test_un.mean(0), ax[1], title="", ylabel="unshift.")
                plotter(t, x_test_br.mean(0), ax[2], title="", xlabel="time (s)", ylabel="shift.")
                plt.tight_layout(rect=[0, 0.05, 1, 0.91])
                LOGGER.info("Saving noisy signals plot.")
                for ext in plot_exts:
                    save_name = f"xn{i}_sAmp{s_amp}_nAmp{n_amp}{ext}"
                    plt.savefig(os.path.join(root, save_name))

                # FFT zašumělých signálů
                fss = np.arange(nfft // 2) / nfft * fs
                X_train = abs(np.fft.fft(x_train, nfft))[:, :nfft // 2]
                X_test_un = abs(np.fft.fft(x_test_un, nfft))[:, :nfft // 2]
                X_test_br = abs(np.fft.fft(x_test_br, nfft))[:, :nfft // 2]
                LOGGER.info("Plotting fft of noisy signals")
                _, ax = plt.subplots(3, 1)
                plt.suptitle("Fourier spectra of noisy signals"
                             f"\n signal: {s_amp}, noise: {n_amp}")
                plotter(freq_train, X_train.mean(0), ax[0], ylabel="abs(train)")
                plotter(freq_test_un, X_test_un.mean(0), ax[1], ylabel="abs(unshift.)")
                plotter(freq_test_br, X_test_br.mean(0), ax[2], xlabel="frequency (Hz)", ylabel="abs(shift.)")
                plt.tight_layout(rect=[0, 0.05, 1, 0.91])
                LOGGER.info("Saving fft of noisy signals plot.")
                for ext in plot_exts:
                    save_name = f"fft{i}_sAmp{s_amp}_nAmp{n_amp}{ext}"
                    plt.savefig(os.path.join(root, save_name))

                # Preprocessing zašumělých signálů a jejich průměr
                LOGGER.info("Plotting mean preprocessed spectral densities of noisy signals")
                _, ax = plt.subplots(3, 1)
                plt.suptitle("Mean preprocessed spectral densities of noisy signals"
                             f"\n signal: {s_amp}, noise: {n_amp}")
                plotter(freq_train, psd_train.mean(axis=(0, -1)), ax[0], ylabel="train")
                plotter(freq_test_un, psd_test_un.mean(axis=(0, -1)), ax[1], ylabel="unshift.")
                plotter(freq_test_br, psd_test_br.mean(axis=(0, -1)), ax[2], xlabel="frequency (Hz)",
                        ylabel="shift.")
                plt.tight_layout(rect=[0, 0.03, 1, 0.91])
                LOGGER.info("Saving mean preprocessed spectral densities of noisy signals plot.")
                for ext in plot_exts:
                    save_name = f"PSD{i}_sAmp{s_amp}_nAmp{n_amp}{ext}"
                    plt.savefig(os.path.join(root, save_name))

                # Vykreslení a uložení grafů jednotlivých signálů
                if plot_divided:
                    LOGGER.info("Plotting divided signals.")
                    LOGGER.info("\t Creating subdirectories")
                    subdirs = ("/divided/raw/training",
                               "/divided/raw/test_unshifted",
                               "/divided/raw/test_shifted")
                    os.makedirs(f"{root}{subdirs[0]}", exist_ok=True)
                    os.makedirs(f"{root}{subdirs[1]}", exist_ok=True)
                    os.makedirs(f"{root}{subdirs[2]}", exist_ok=True)
                    LOGGER.info("\t Training signals")
                    _, ax = plt.subplots(1, 3)
                    for j in range(3):
                        plotter(t, x_train[j, :], ax[j], title=f"No. {j}", xlabel="time (s)")
                    plt.suptitle("Training signals")
                    ax[0].set_ylabel("day 1")
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    for ext in plot_exts:
                        plt.savefig(f"{root}{subdirs[0]}/xn{i}_sAmp{s_amp}_nAmp{n_amp}{ext}")

                    LOGGER.info("\t Unshifted test signals")
                    _, ax = plt.subplots(2, 3)
                    ax = ax.flatten()
                    for j in range(6):
                        plotter(t, x_test_un[j, :], ax[j], title=f"No. {j}", xlabel="time (s)")
                    plt.suptitle("Unshifted test signals")
                    ax[0].set_ylabel("day 2")
                    ax[3].set_ylabel("day 3")
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    for ext in plot_exts:
                        plt.savefig(f"{root}{subdirs[1]}/xn{i}_sAmp{s_amp}_nAmp{n_amp}{ext}")

                    LOGGER.info("\t Shifted test signals")
                    _, ax = plt.subplots(2, 3)
                    ax = ax.flatten()
                    for j in range(6):
                        plotter(t, x_test_br[j, :], ax[j], title=f"No. {j}", xlabel="time (s)")
                    plt.suptitle("Shifted test signals")
                    ax[0].set_ylabel("day 4")
                    ax[3].set_ylabel("day 5")
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    for ext in plot_exts:
                        plt.savefig(f"{root}{subdirs[2]}/xn{i}_sAmp{s_amp}_nAmp{n_amp}{ext}")

                    LOGGER.info("Plotting divided FFTs.")
                    LOGGER.info("\t Creating subdirectories")
                    subdirs = ("/divided/fft/training",
                               "/divided/fft/test_unshifted",
                               "/divided/fft/test_shifted")
                    os.makedirs(f"{root}{subdirs[0]}", exist_ok=True)
                    os.makedirs(f"{root}{subdirs[1]}", exist_ok=True)
                    os.makedirs(f"{root}{subdirs[2]}", exist_ok=True)
                    LOGGER.info("\t Training FFTs")
                    _, ax = plt.subplots(1, 3)
                    for j in range(3):
                        plotter(freq_train, X_train[j, :], ax[j], title=f"No. {j}", xlabel="frequency (Hz)")
                    plt.suptitle("Training FFTs")
                    ax[0].set_ylabel("day 1")
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    for ext in plot_exts:
                        plt.savefig(f"{root}{subdirs[0]}/fft{i}_sAmp{s_amp}_nAmp{n_amp}{ext}")

                    LOGGER.info("\t Unshifted test FFTs")
                    _, ax = plt.subplots(2, 3)
                    ax = ax.flatten()
                    for j in range(6):
                        plotter(freq_test_un, X_test_un[j, :], ax[j], title=f"No. {j}", xlabel="frequency (Hz)")
                    plt.suptitle("Unshifted test FFTs")
                    ax[0].set_ylabel("day 2")
                    ax[3].set_ylabel("day 3")
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    for ext in plot_exts:
                        plt.savefig(f"{root}{subdirs[1]}/fft{i}_sAmp{s_amp}_nAmp{n_amp}{ext}")

                    LOGGER.info("\t Shifted test FFTs")
                    _, ax = plt.subplots(2, 3)
                    ax = ax.flatten()
                    for j in range(6):
                        plotter(freq_test_br, X_test_br[j, :], ax[j], title=f"No. {j}", xlabel="frequency (Hz)")
                    plt.suptitle("Shifted test FFTs")
                    ax[0].set_ylabel("day 4")
                    ax[3].set_ylabel("day 5")
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    for ext in plot_exts:
                        plt.savefig(f"{root}{subdirs[2]}/fft{i}_sAmp{s_amp}_nAmp{n_amp}{ext}")

                    LOGGER.info("Plotting divided PSDs.")
                    LOGGER.info("\t Creating subdirectories")
                    subdirs = ("/divided/psd/training",
                               "/divided/psd/test_unshifted",
                               "/divided/psd/test_shifted")
                    os.makedirs(f"{root}{subdirs[0]}", exist_ok=True)
                    os.makedirs(f"{root}{subdirs[1]}", exist_ok=True)
                    os.makedirs(f"{root}{subdirs[2]}", exist_ok=True)
                    LOGGER.info("\t Training PSDs")
                    _, ax = plt.subplots(1, 3)
                    for j in range(3):
                        plotter(freq_train, psd_train[j, :], ax[j], title=f"No. {j}", xlabel="frequency (Hz)")
                    plt.suptitle("Training PSDs")
                    ax[0].set_ylabel("day 1")
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    for ext in plot_exts:
                        plt.savefig(f"{root}{subdirs[0]}/PSD{i}_sAmp{s_amp}_nAmp{n_amp}{ext}")

                    LOGGER.info("\t Unshifted test PSDs")
                    _, ax = plt.subplots(2, 3)
                    ax = ax.flatten()
                    for j in range(6):
                        plotter(freq_test_un, psd_test_un[j, :], ax[j], title=f"No. {j}", xlabel="frequency (Hz)")
                    plt.suptitle("Unshifted test PSDs")
                    ax[0].set_ylabel("day 2")
                    ax[3].set_ylabel("day 3")
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    for ext in plot_exts:
                        plt.savefig(f"{root}{subdirs[1]}/PSD{i}_sAmp{s_amp}_nAmp{n_amp}{ext}")

                    LOGGER.info("\t Shifted test PSDs")
                    _, ax = plt.subplots(2, 3)
                    ax = ax.flatten()
                    for j in range(6):
                        plotter(freq_test_br, psd_test_br[j, :], ax[j], title=f"No. {j}", xlabel="frequency (Hz)")
                    plt.suptitle("Shifted test PSDs")
                    ax[0].set_ylabel("day 4")
                    ax[3].set_ylabel("day 5")
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    for ext in plot_exts:
                        plt.savefig(f"{root}{subdirs[2]}/PSD{i}_sAmp{s_amp}_nAmp{n_amp}{ext}")


                # Uložení PSNR do slovníku
                # {"i": [], "signal_amp": [], "noise_amp": [], "train_PSNR (dB)": [], "test_PSNR (dB)": []}
                LOGGER.info("Saving PSNR values to dictionary")
                dPSNR["episode"].append(i)
                dPSNR["signal_amp_max"].append(s_amp[1])
                dPSNR["noise_amp_max"].append(n_amp[1])
                dPSNR["train_PSNR (dB)"].append(PSNR_train)
                dPSNR["test_PSNR (dB)"].append(np.mean([PSNR_test_un, PSNR_test_br]))
                LOGGER.debug(f"dPSNR.values: {list(dPSNR.values())}")
    dfPSNR = pd.DataFrame(dPSNR, columns=list(dPSNR.keys()))
    with open(os.path.join(root, PSNR_CSV_SETUP["name"]), "w") as f:
        dfPSNR.to_csv(f, sep=PSNR_CSV_SETUP["sep"], decimal=PSNR_CSV_SETUP["decimal"], index=PSNR_CSV_SETUP["index"],
                      columns=PSNR_CSV_SETUP["columns"], line_terminator=PSNR_CSV_SETUP["line_terminator"])

    plt.show()
