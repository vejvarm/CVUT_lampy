import numpy as np

from dev.helpers import console_logger
from dev.Generator import Generator
from Methods import M2
from preprocessing import Preprocessor

LOGGER = console_logger(__name__, "DEBUG")

if __name__ == '__main__':
    fs = 512
    nsamples = 15360

    dtrain = {"nsig": 100, "fvs": (5., 10., 30., 80., 110.), "amp_range": (1, 5), "shift_range": (-1, 1)}
    dtest = {"nsig": 120, "fvs": (8., 7., 33., 77., 113.), "amp_range": (1, 5), "shift_range": (-1, 1)}
    dnoise = {"fvs": np.arange(fs//2), "amp_range": (0, 3), "shift_range": (0, 0)}

    LOGGER.info("Initialising signal and noise generators.")
    g_train = Generator(fs, fvs=dtrain["fvs"], amp_range=dtrain["amp_range"], shift_range=dtrain["shift_range"])
    g_test = Generator(fs, fvs=dtest["fvs"], amp_range=dtest["amp_range"], shift_range=dtest["shift_range"])
    g_noise = Generator(fs, fvs=dnoise["fvs"], amp_range=dnoise["amp_range"], shift_range=dnoise["shift_range"])

    LOGGER.info("Generating signal values with added noise.")
    t = np.arange(nsamples)/fs
    x_train = g_train.generate(dtrain["nsig"], nsamples) + g_noise.generate(dtrain["nsig"], nsamples)
    x_test = g_test.generate(dtest["nsig"], nsamples) + g_noise.generate(dtest["nsig"], nsamples)
    LOGGER.debug(f"x_train.shape: {x_train.shape}, x_test.shape: {x_test.shape}")

    LOGGER.info("Preprocessing signals to psd using Preprocessor with default values.")
    p = Preprocessor()
    freq_train, psd_train = p.simple_preprocess(x_train.T)
    freq_test, psd_test = p.simple_preprocess(x_test.T)
    LOGGER.debug(f"psd_train.shape: {psd_train.shape}, psd_test.shape: {psd_test.shape}")

    LOGGER.info("Instantiating Multiscale cross entropy method (M2)")
    m2 = M2()

    # TODO: train M2 on psd_train
    # TODO: compare trained binarized spectra with testing binarized spectra

