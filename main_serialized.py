import os

from collections import Counter

from bin.flags import FLAGS
from bin.Preprocessor import Preprocessor
from bin.Methods import M2

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# TODO: korelace mezi sílou větru, rychlostí větru a amplitudou (vybuzeností) vlastních frekvencí

FULL_IMAGE_SAVE_FOLDER = os.path.join(FLAGS.data_root, FLAGS.image_save_folder)
PRINT_FILL_SIZE = 50

def calc_periodic_best(ce, bin_sizes, thresholds, print_results=False):
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
        if print_results:
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
    root = os.path.join(FLAGS.data_root, FLAGS.preprocessed_folder)
    # Paths to training files
    setting = "training"
    folder = FLAGS.paths[setting]["folder"]
    dataset = FLAGS.paths[setting]["dataset"]
    period = [FLAGS.paths[setting]["period"]]*len(dataset)
    paths = [os.path.join(root, folder, d, p) for d, p in zip(dataset, period)]

    os.makedirs(FULL_IMAGE_SAVE_FOLDER, exist_ok=True)

    # Paths to validation files
    setting_valid = "serialized"
    folder = FLAGS.paths[setting_valid]["folder"]
    dataset = FLAGS.paths[setting_valid]["dataset"]
    period = [FLAGS.paths[setting_valid]["period"]]*len(dataset)
    paths_valid = [os.path.join(root, folder, d, p) for d, p in zip(dataset, period)]

    from_existing_file = True

    # TODO: Make grid search for lamp, bin_sizes, thresholds and period
    # TODO: Save and categorize resulting graphs

    for lamp in FLAGS.LAMPS_GRID:
        # relative difference container
        df = pd.DataFrame({"lamp": [],
                           "bin_sizes": [],
                           "thresholds": [],
                           "per": [],
                           "vs": [],
                           "da": []})
        for bin_sizes in FLAGS.BIN_SIZE_GRID:
            for thresholds in FLAGS.THRESHOLD_GRID:
                for period in FLAGS.PERIOD_GRID:
                    for var_scaled in FLAGS.VAR_SCALED_GRID:
                        print(" SETUP ".center(PRINT_FILL_SIZE, "#"))
                        print(f"lamp: {lamp}".center(PRINT_FILL_SIZE, " "))
                        print(f"bsz: {bin_sizes}".center(PRINT_FILL_SIZE, " "))
                        print(f"ths: {thresholds}".center(PRINT_FILL_SIZE, " "))
                        print(f"per: {period} | vs: {var_scaled}".center(PRINT_FILL_SIZE, " "))
                        print("".center(PRINT_FILL_SIZE, "-"))
                        # multiscale params
                        # bin_sizes = (8, )
                        # thresholds = (0.01, )
                        plot_distributions = False

                        if len(bin_sizes) * len(thresholds) == 1:
                            method_name = "CCE"
                        else:
                            method_name = "MCCE"

                        # periodic params
                        # period = 1
                        ndays_unbroken = FLAGS.serialized["unbroken"]//period
                        ndays_broken = FLAGS.serialized["broken"]//period
                        ndays = ndays_unbroken + ndays_broken


                        # define instance of Preprocessor and initialize M2
                        preprocessor = Preprocessor()
                        m2 = M2(preprocessor, from_existing_file=from_existing_file, from_preprocessed=True, lamp=lamp, var_scaled_PSD=var_scaled)

                        # Train the method on 2 months of neporuseno (trained)
                        m2.train(paths[0], bin_sizes, thresholds)

                        # Calculate cross entropy of
                        ce23 = m2.compare(paths_valid[0], period=period, print_results=False)  # compared with neporuseno2
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
                        rel_diff = (a3 - a2) / a2 * 100
                        data_row = pd.DataFrame({"lamp": [lamp],
                                                 "bin_sizes": [bin_sizes],
                                                 "thresholds": [thresholds],
                                                 "per": [period],
                                                 "vs": [var_scaled],
                                                 "da": [rel_diff]})
                        df = df.append(data_row, ignore_index=True)
                        print(data_row)

                        # plot the results of cummulative cross-entropies and their regression
                        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
                        plt.plot(x23, a2*x23 + b2, "b", label=f"$r_h$ (α={a2:.1f})")
                        plt.plot(x23, a3*x23 + b3, "r", label=f"$r_d$ (α={a3:.1f})")
                        plt.stem(y2, markerfmt="bx", linefmt="none", basefmt=" ", use_line_collection=True,
                                 label=f"{method_name} of $PSD_h$")
                        plt.stem(x3, y3, markerfmt="r+", linefmt="none", basefmt=" ", use_line_collection=True,
                                 label=f"{method_name} of $PSD_d$")
                        plt.ylim([0, None])
                        plt.xlabel(f"period ({period} " + ("day" if period == 1 else "days") + ")")
                        plt.ylabel(f"{method_name}")
                        plt.title(f"{method_name} for {lamp.upper()} | bs: {bin_sizes} | th: {thresholds} | dα: {rel_diff:.2f} %)")
                        plt.legend()
                        plt.grid()

                        # save the resulting plot
                        flnm = os.path.join(FULL_IMAGE_SAVE_FOLDER, f"{method_name}_{lamp}_bs{bin_sizes}_th{thresholds}_per{period}_vs{var_scaled}")
                        plt.savefig(f"{flnm}.png", dpi=200)
                        plt.savefig(f"{flnm}.pdf")
                        plt.savefig(f"{flnm}.svg")
                        # close the figure
                        plt.close(fig)
                        # plt.show()

                        print(" DONE ".center(PRINT_FILL_SIZE, "#")+"\n")

        df.to_csv(os.path.join(FULL_IMAGE_SAVE_FOLDER, f"da_{lamp}_results.csv"))