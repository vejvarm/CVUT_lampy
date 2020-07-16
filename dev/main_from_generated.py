import os
from collections import Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from bin.flags import FLAGS
from bin.helpers import console_logger
from bin.Preprocessor import Preprocessor
from bin.Methods import M2

# TODO: korelace mezi sílou větru, rychlostí větru a amplitudou (vybuzeností) vlastních frekvencí

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
    signal_amps = [(0.75, 1)]
    noise_amps = [(0, 0.5)]
    root = "../data/for_article/generated"

    # PARAMS
    from_existing_file = True

    # multiscale params
    bin_sizes = (32, 48, )
    thresholds = (0.5, 2.5)
    plot_distributions = False

    # periodic params
    period = 1
    nmeas = 3  # number of measurements in one day!
    ndays = 4

    ndays_unbroken = 2
    ndays_broken = 2

    # relative difference container
    rel_diff_list = []

    # PSNR
    with open(os.path.join(root, PSNR_CSV_SETUP["name"]), "r") as f:
        dfPSNR = pd.read_csv(f, sep=PSNR_CSV_SETUP["sep"], decimal=PSNR_CSV_SETUP["decimal"],
                             index_col=PSNR_CSV_SETUP["index"], lineterminator=PSNR_CSV_SETUP["line_terminator"])

    # Paths to files
    for idx in range(nrepeats):
        for s_amp in signal_amps:
            for n_amp in noise_amps:
                path_train = {"folder": f"{root}/train", "name": f"X{idx}_sAmp{s_amp}_nAmp{n_amp}.npy"}
                path_test = {"folder": f"{root}/test", "name": f"X{idx}_sAmp{s_amp}_nAmp{n_amp}.npy"}

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

                # get row from PSNR dataframe for current data
                row = dfPSNR.loc[(dfPSNR[_EPISODE] == idx) &
                                 (dfPSNR[_SIGNAL_AMP] == s_amp[1]) &
                                 (dfPSNR[_NOISE_AMP] == n_amp[1])]
                psnr = (float(row[_TRAIN_PSNR]) + float(row[_TEST_PSNR]))/2
                LOGGER.debug(f"PSNR value: {psnr}")

                # plot the results of cummulative cross-entropies and their regression
                LOGGER.info("Plotting resulting MCCE")
                fig = plt.figure()
                plt.plot(x23, a2*x23 + b2, "b", label=f"regress. unshifted frequencies (α={a2:.1f})")
                plt.plot(x23, a3*x23 + b3, "r", label=f"regress. shifted frequencies (α={a3:.1f})")
                plt.stem(y2, markerfmt="bx", linefmt="none", basefmt=" ", use_line_collection=True, label="MCCE of freq. unshifted signals")
                plt.stem(x3, y3, markerfmt="r+", linefmt="none", basefmt=" ", use_line_collection=True, label="MCCE of freq. shifted signals")
                plt.ylim([0, None])
                plt.xlabel(f"days (period = {period} " + ("day" if period == 1 else "days") + ")")
                plt.ylabel("MCCE")
                plt.title(f"MCCE with regression \n (PSNR: {psnr:.2f} dB | dα: {rel_diff:.2f} %)")
                plt.legend()
                plt.grid()

                # save the resulting plot
                LOGGER.info("Saving current plot")
                flnm = f"../images/M2/cce_nd-{ndays}_p-{period}_i_{idx}_sAmp{s_amp}_nAmp{n_amp}"
                plt.savefig(f"{flnm}.png", dpi=200)
                plt.savefig(f"{flnm}.pdf")
                plt.savefig(f"{flnm}.svg")

                # if plot distributions:
                if plot_distributions:
                    LOGGER.info("Plotting trained binarized distributions")
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
                            flnm = f"../images/M2/binarized-spectra/binarized_spectrum_bs-{params[0]}_th-{params[1]}"
                            plt.savefig(f"{flnm}.png", dpi=200)
                            plt.savefig(f"{flnm}.pdf")
                            plt.savefig(f"{flnm}.svg")
    LOGGER.info(f"Adding dα column to dfPSNR")
    dfPSNR["da (%)"] = pd.Series(rel_diff_list, index=dfPSNR.index)
    LOGGER.info(f"Saving updated dfPSNR")
    with open(os.path.join(root, PSNR_CSV_SETUP["name2"]), "w") as f:
        dfPSNR.to_csv(f, sep=PSNR_CSV_SETUP["sep"], decimal=PSNR_CSV_SETUP["decimal"], index=PSNR_CSV_SETUP["index"],
                      line_terminator=PSNR_CSV_SETUP["line_terminator"])
    plt.show()