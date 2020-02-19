import os

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from flags import FLAGS
from dev.helpers import console_logger, plotter
from dev.Generator import Generator
from preprocessing import Preprocessor

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


if __name__ == '__main__':
    nrepeats = 10
    signal_amps = [(0, 1)]
    noise_amps = [(0, 0), (0, 0.25), (0, 0.5), (0, 0.75), (0, 1), (0, 1.5), (0, 2), (0, 4)]

    fs = 512
    nfft = 5120
    nsamples = 15360
    root = "../data/generated"

    dPSNR = {"episode": [], "signal_amp_max": [], "noise_amp_max": [], "train_PSNR (dB)": [], "test_PSNR (dB)": []}

    for i in range(nrepeats):
        for s_amp in signal_amps:
            for n_amp in noise_amps:
                dtrain = {"nsig": 10, "fvs": (5., 10., 30., 80., 110.), "amp_range": s_amp,
                          "shift_range": (0, 0), "path": f"{root}/train/"}
                dtest = {"nsig": 20, "fvs": (5., 10., 30., 80., 110.), "amp_range": s_amp,
                         "shift_range": (0, 0), "path": f"{root}/test/"}
                dtest_br = {"nsig": 20, "fvs": (8., 7., 33., 77., 113.), "amp_range": s_amp,
                            "shift_range": (0, 0), "path": f"{root}/test/"}
                dnoise = {"fvs": np.arange(fs//2), "amp_range": n_amp, "shift_range": (0, 0)}

                LOGGER.info("Initialising signal and noise generators.")
                g_train = Generator(fs, fvs=dtrain["fvs"], amp_range=dtrain["amp_range"], shift_range=dtrain["shift_range"])
                g_test = Generator(fs, fvs=dtest["fvs"], amp_range=dtest["amp_range"], shift_range=dtest["shift_range"])
                g_test_br = Generator(fs, fvs=dtest_br["fvs"], amp_range=dtest_br["amp_range"], shift_range=dtest_br["shift_range"])
                g_noise = Generator(fs, fvs=dnoise["fvs"], amp_range=dnoise["amp_range"], shift_range=dnoise["shift_range"])

                LOGGER.info("Generating signal values with added noise.")
                t = np.arange(nsamples)/fs
                x_train, PSNR_train = generate_and_compute_snr(g_train, g_noise, shape=(dtrain["nsig"], nsamples))
                x_test, PSNR_test = generate_and_compute_snr(g_test, g_noise, shape=(dtest["nsig"], nsamples))
                x_test_br, PSNR_test_br = generate_and_compute_snr(g_test_br, g_noise, shape=(dtest_br["nsig"], nsamples))
                LOGGER.debug(f"x_train.shape: {x_train.shape}, x_test.shape: {x_test.shape}, x_test_br.shape: {x_test_br.shape}")

                LOGGER.info("Concatenating unbroken and broken test data")
                x_test = np.vstack([x_test, x_test_br])

                LOGGER.info("Preprocessing signals to psd using Preprocessor with default values.")
                p = Preprocessor()  # TODO: rem_neg (yes or no?)
                freq_train, psd_train = p.simple_preprocess(x_train.T)
                freq_test, psd_test = p.simple_preprocess(x_test.T)
                LOGGER.debug(f"psd_train.shape: {psd_train.shape}, psd_test.shape: {psd_test.shape}")

                LOGGER.info("Reshaping to M2 compatible shape.")
                psd_train = np.expand_dims(psd_train.T, axis=-1)
                psd_test = np.expand_dims(psd_test.T, axis=-1)
                LOGGER.debug(f"psd_train.shape: {psd_train.shape}, psd_test.shape: {psd_test.shape}")

                LOGGER.debug("Making folder if it doesn't exist")
                os.makedirs(dtrain["path"], exist_ok=True)
                os.makedirs(dtest["path"], exist_ok=True)

                LOGGER.info("Generating save file name.")
                save_name = f"X{i}_sAmp{s_amp}_nAmp{n_amp}.npy"
                freqs_name = f"freqs{i}.npy"
                LOGGER.info("Saving generated arrays to files.")
                np.save(os.path.join(dtrain["path"], "freqs.npy"), freq_train)
                LOGGER.debug(f"freq_train saved to {dtrain['path']} with file name {freqs_name}")
                np.save(os.path.join(dtrain["path"], save_name), psd_train)
                LOGGER.debug(f"psd_train saved to {dtrain['path']} with file name {save_name}")
                np.save(os.path.join(dtest["path"], "freqs.npy"), freq_test)
                LOGGER.debug(f"freq_test saved to {dtest['path']} with file name {freqs_name}")
                np.save(os.path.join(dtest["path"], save_name), psd_test)
                LOGGER.debug(f"psd_test saved to {dtest['path']} with file name {save_name}")

                # Vykreslení výsledků
                LOGGER.info("Plotting results.")
                _, ax = plt.subplots(2, 1)
                LOGGER.info("Plotting noisy signals.")
                plt.suptitle("Noisy signals"
                             f"\n signal: {s_amp}, noise: {n_amp}")
                plotter(t, x_train.mean(0), ax[0], title="", ylabel="training signal")
                plotter(t, x_test.mean(0), ax[1], title="", xlabel="time (s)", ylabel="testing signal")
                LOGGER.info("Saving noisy signals plot.")
                save_name = f"xn{i}_sAmp{s_amp}_nAmp{n_amp}.pdf"
                plt.savefig(os.path.join(root, save_name))

                # FFT zašumělých signálů
                fss = np.arange(nfft // 2) / nfft * fs
                X_train = abs(np.fft.fft(x_train, nfft))[:, :nfft // 2]
                X_test = abs(np.fft.fft(x_test, nfft))[:, :nfft // 2]
                LOGGER.info("Plotting fft of noisy signals")
                _, ax = plt.subplots(2, 1)
                plt.suptitle("Fourier spectra of noisy signals"
                             f"\n signal: {s_amp}, noise: {n_amp}")
                plotter(freq_train, X_train.mean(0), ax[0], ylabel="abs(fft_train)")
                plotter(freq_test, X_test.mean(0), ax[1], xlabel="frequency (Hz)", ylabel="abs(fft_test)")
                LOGGER.info("Saving fft of noisy signals plot.")
                save_name = f"fft{i}_sAmp{s_amp}_nAmp{n_amp}.pdf"
                plt.savefig(os.path.join(root, save_name))

                # Preprocessing zašumělých signálů a jejich průměr
                LOGGER.info("Plotting mean preprocessed spectral densities of noisy signals")
                _, ax = plt.subplots(2, 1)
                plt.suptitle("Mean preprocessed spectral densities of noisy signals"
                             f"\n signal: {s_amp}, noise: {n_amp}")
                plotter(freq_train, psd_train.mean(axis=(0, -1)), ax[0], ylabel="psd_train")
                plotter(freq_test, psd_test.mean(axis=(0, -1)), ax[1], xlabel="frequency (Hz)",
                        ylabel="psd_test")
                LOGGER.info("Saving mean preprocessed spectral densities of noisy signals plot.")
                save_name = f"PSD{i}_sAmp{s_amp}_nAmp{n_amp}.pdf"
                plt.savefig(os.path.join(root, save_name))

                # Uložení PSNR do slovníku
                # {"i": [], "signal_amp": [], "noise_amp": [], "train_PSNR (dB)": [], "test_PSNR (dB)": []}
                LOGGER.info("Saving PSNR values to dictionary")
                dPSNR["episode"].append(i)
                dPSNR["signal_amp_max"].append(s_amp[1])
                dPSNR["noise_amp_max"].append(n_amp[1])
                dPSNR["train_PSNR (dB)"].append(PSNR_train)
                dPSNR["test_PSNR (dB)"].append(np.mean([PSNR_test, PSNR_test_br]))
                LOGGER.debug(f"dPSNR.values: {list(dPSNR.values())}")
    dfPSNR = pd.DataFrame(dPSNR, columns=list(dPSNR.keys()))
    with open(os.path.join(root, PSNR_CSV_SETUP["name"]), "w") as f:
        dfPSNR.to_csv(f, sep=PSNR_CSV_SETUP["sep"], decimal=PSNR_CSV_SETUP["decimal"], index=PSNR_CSV_SETUP["index"],
                      columns=PSNR_CSV_SETUP["columns"], line_terminator=PSNR_CSV_SETUP["line_terminator"])
    plt.show()
