import os

import numpy as np

from dev.helpers import console_logger
from dev.Generator import Generator
from Methods import M2
from preprocessing import Preprocessor

LOGGER = console_logger(__name__, "DEBUG")

if __name__ == '__main__':
    nrepeats = 1
    signal_amps = [(4, 5)]
    noise_amps = [(0, 1), (0, 2), (0, 3), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5),
                  (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11)]

    fs = 512
    nsamples = 15360

    for i in range(nrepeats):
        for s_amp in signal_amps:
            for n_amp in noise_amps:
                dtrain = {"nsig": 10, "fvs": (5., 10., 30., 80., 110.), "amp_range": s_amp,
                          "shift_range": (0, 0), "path": "../data/generated/train/"}
                dtest = {"nsig": 20, "fvs": (5., 10., 30., 80., 110.), "amp_range": s_amp,
                         "shift_range": (0, 0), "path": "../data/generated/test/"}
                dtest_br = {"nsig": 20, "fvs": (8., 7., 33., 77., 113.), "amp_range": s_amp,
                            "shift_range": (0, 0), "path": "../data/generated/test/"}
                dnoise = {"fvs": np.arange(fs//2), "amp_range": n_amp, "shift_range": (0, 0)}

                LOGGER.info("Initialising signal and noise generators.")
                g_train = Generator(fs, fvs=dtrain["fvs"], amp_range=dtrain["amp_range"], shift_range=dtrain["shift_range"])
                g_test = Generator(fs, fvs=dtest["fvs"], amp_range=dtest["amp_range"], shift_range=dtest["shift_range"])
                g_test_br = Generator(fs, fvs=dtest_br["fvs"], amp_range=dtest_br["amp_range"], shift_range=dtest_br["shift_range"])
                g_noise = Generator(fs, fvs=dnoise["fvs"], amp_range=dnoise["amp_range"], shift_range=dnoise["shift_range"])

                LOGGER.info("Generating signal values with added noise.")
                t = np.arange(nsamples)/fs
                x_train = g_train.generate(dtrain["nsig"], nsamples) + g_noise.generate(dtrain["nsig"], nsamples)
                x_test = g_test.generate(dtest["nsig"], nsamples) + g_noise.generate(dtest["nsig"], nsamples)
                x_test_br = g_test_br.generate(dtest_br["nsig"], nsamples) + g_noise.generate(dtest_br["nsig"], nsamples)
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

