import os
from collections import Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from flags import FLAGS
from dev.helpers import console_logger
from preprocessing import Preprocessor
from Methods import M2

LOGGER = console_logger(__name__, "DEBUG")
PSNR_CSV_SETUP = FLAGS.PSNR_csv_setup

_EPISODE = PSNR_CSV_SETUP["columns"][0]
_SIGNAL_AMP = PSNR_CSV_SETUP["columns"][1]
_NOISE_AMP = PSNR_CSV_SETUP["columns"][2]
_TRAIN_PSNR = PSNR_CSV_SETUP["columns"][-2]
_TEST_PSNR = PSNR_CSV_SETUP["columns"][-1]


def calc_periodic_best(ce, bin_sizes, thresholds):
    nperiods, nparams = ce.shape
    nth = len(thresholds)

    best_params = [np.nan]*nperiods
    periodic_best = np.empty((nperiods,), dtype=np.float32)
    periodic_best[:] = -np.inf
    for i, day in enumerate(ce):
        best_j = 0
        for j, val in enumerate(day):
            if val > periodic_best[i]:
                periodic_best[i] = val
                best_j = j
        best_params[i] = divmod(best_j, nth)
        print(f"day: {i}, bs: {bin_sizes[best_params[i][0]]}, th: {thresholds[best_params[i][1]]} val: {periodic_best[i]}")
    return best_params, periodic_best


def linear_regression(y, x=None):
    """Calculate params of a line (a, b) using least squares algorithm to best fit the input data (x, y)"""
    ndata = len(y)
    if not all(x):
        x = np.arange(0, ndata, 1)
    else:
        assert len(x) == ndata, "x and y don't have the same length"
    A = np.vstack((x, np.ones(ndata))).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return a, b


if __name__ == "__main__":
    # DATA LOADING SETTINGS
    nrepeats = 1
    signal_amps = [(0, 1)]
    noise_amps = [(0, 0)]
    bearing = "DE"  # "FE" or "DE"
    fault = "IR"  # "IR", "BL", "ORat06", "ORat03", "ORat12"
    if "DE" in bearing:
        root = "../data/drive_end/"
    elif "FE" in bearing:
        root = "../data/fan_end/"
    else:
        raise(ValueError, "Bearing choice must be either Drive End ('DE') or Fan End ('FE')")

    if "IR" in fault:
        root = root+"IR/"
    elif "ORat06" in fault:
        root = root+"ORat06/"
    elif "ORat03" in fault:
        root = root+"ORat03/"
    elif "ORat12" in fault:
        root = root+"ORat12/"
    elif "BL" in fault:
        root = root+"BL/"
    else:
        raise(ValueError, "Fault choice must be one of Inner Rail ('IR'), Outer Rail ('ORat##'), or  Ball ('BL')")

    # PARAMS
    from_existing_file = True

    # multiscale params
    bin_sizes = (5, 10, 20)
    thresholds = (0.1, 0.5, 0.8)
    plot_distributions = False

    # periodic params
    period = 1
    nmeas = 1  # number of measurements in one day!
    ndays = 12

    ndays_unbroken = 4
    ndays_broken = 8

    # relative difference container
    rel_diff_list = []

    # Paths to files
    for idx in range(nrepeats):
        for s_amp in signal_amps:
            for n_amp in noise_amps:
                path_train = {"folder": f"{root}/train", "name": f"X.npy"}
                path_test = {"folder": f"{root}/test", "name": f"X.npy"}

                # define instance of Preprocessor and initialize M2
                preprocessor = Preprocessor()
                m2 = M2(preprocessor, from_existing_file=from_existing_file, nmeas=nmeas)

                # Train the method on 2 months of neporuseno (trained)
                m2.train(os.path.join(*list(path_train.values())), bin_sizes, thresholds)

                # Calculate cross entropy of
                ce23 = m2.compare(os.path.join(*list(path_test.values())), period=period, print_results=False)  # compared with neporuseno2
                # print(ce2.shape)  # (nperiods, nbins*nthresholds)

                # Find the highest cross-entropy for each day
                ce23_best_params, ce23_periodic_best = calc_periodic_best(ce23, bin_sizes, thresholds)

                # Count which and how many times have combinations of (bin_size, threshold) been chosen as highest ce
                ce23_best_js = Counter(ce23_best_params)

                # Calculate cummulative sum of cross-entropies
                x23 = np.arange(0, len(ce23_periodic_best), 1)[:ndays]
                x2 = x23[:ndays_unbroken]
                x3 = x23[ndays_unbroken:]
                y23 = np.cumsum(ce23_periodic_best)[:ndays]
                y2 = y23[:ndays_unbroken]
                y3 = y23[ndays_unbroken:]

                # Calculate params for linear regressions
                a2, b2 = linear_regression(y2, x2)
                a3, b3 = linear_regression(y3, x3)
                rel_diff = (a3 - a2)/a2*100
                rel_diff_list.append(rel_diff)

                # plot the results of cummulative cross-entropies and their regression
                LOGGER.info("Plotting resulting MCCE")
                fig = plt.figure()
                plt.plot(x23, a2*x23 + b2, "b", label=f"regress. unshifted frequencies (α={a2:.1f})")
                plt.plot(x23, a3*x23 + b3, "r", label=f"regress. shifted frequencies (α={a3:.1f})")
                plt.stem(y2, markerfmt="bx", linefmt="none", basefmt=" ", use_line_collection=True, label="MCCE of freq. unshifted signals")
                plt.stem(x3, y3, markerfmt="r+", linefmt="none", basefmt=" ", use_line_collection=True, label="MCCE of freq. shifted signals")
                plt.ylim([0, None])
                plt.xlabel(f"period ({period} " + ("signal" if period == 1 else "signals") + ")")
                plt.ylabel("MCCE")
                plt.title(f"MCCE with regression \n (bearing: {bearing} | fault: {fault} | dα: {rel_diff:.2f} %)")
                plt.legend()
                plt.grid()

                # save the resulting plot
                LOGGER.info("Saving current plot")
                os.makedirs(f"{root}/images", exist_ok=True)
                plt.savefig(f"{root}/images/cce_nd-{ndays}_p-{period}_i_{idx}_sAmp{s_amp}_nAmp{n_amp}.pdf")

                # if plot distributions:
                if plot_distributions:
                    LOGGER.info("Plotting binarized distributions")
                    for j, dist in enumerate((m2.trained_distributions, )):
                        for params, freqs, d in dist:
                            nrows = 1
                            ncols = 1
                            fig, axes = plt.subplots(nrows, ncols)
                            if nrows+ncols == 2:
                                axes = np.array([axes])
                            for i, ax in enumerate(axes.flatten()):
                                y_pos = np.arange(len(d[i, :]))
                                ax.bar(y_pos, d[i, :], align="center", width=0.9)
                                ax.set_xlabel("bin number")
                                ax.set_ylabel("Softmax(psd_binarized) (1)")
                                ax.set_yscale("log")
                            fig.suptitle(f"Binarized spectrum | bin size: {params[0]} | threshold: {params[1]} |")
                            plt.savefig(f"../images/M2/binarized-spectra/binarized_spectrum_bs-{params[0]}_th-{params[1]}.png", dpi=200)
    plt.show()