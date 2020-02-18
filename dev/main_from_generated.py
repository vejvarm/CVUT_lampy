import os
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt

from flags import FLAGS
from dev.helpers import console_logger
from preprocessing import Preprocessor
from Methods import M2

# TODO: korelace mezi sílou větru, rychlostí větru a amplitudou (vybuzeností) vlastních frekvencí

LOGGER = console_logger(__name__, "DEBUG")


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
    # Paths to files
    root = "../data/generated"
    path_train = {"folder": f"{root}/train", "name": "X1.npy"}
    path_test = {"folder": f"{root}/test", "name": "X1.npy"}

    from_existing_file = True

    # multiscale params
    bin_sizes = (40, 80, )
    thresholds = (0.1, 0.8, )
    plot_distributions = True

    # periodic params
    period = 1
    nmeas = 1  # number of measurements in one day!
    ndays = 40

    ndays_unbroken = 20
    ndays_broken = 20


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
    a23, b23 = linear_regression(y23, x23)

    # plot the results of cummulative cross-entropies and their regression
    plt.plot(x23, a2*x23 + b2, "b", label=f"regrese bez porušených dat (a={a2:.1f}, b={b2:.1f})")
    plt.plot(x23, a23*x23 + b23, "r", label=f"regrese s porušenými daty (a={a23:.1f}, b={b23:.1f})")
    plt.stem(y2, markerfmt="bx", linefmt="none", basefmt=" ", use_line_collection=True, label="neporušená lampa")
    plt.stem(x3, y3, markerfmt="r+", linefmt="none", basefmt=" ", use_line_collection=True, label="porušená lampa")
    plt.xlabel(f"perioda ({period} " + ("den" if period == 1 else "dnů") + ")")
    plt.ylabel("kumulativní křížová entropie")
    plt.title("Porovnání kumulativních křížových entropií")
    plt.legend()
    plt.grid()

    # save the resulting plot
    plt.savefig(f"../images/M2/cummul_ce_nd-{ndays}_p-{period}.pdf")

    # if plot distributions:
    if plot_distributions:
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
                    ax.set_xlabel("košík (bin)")
                    ax.set_ylabel("Softmax(psd_binarized) (1)")
                    ax.set_yscale("log")
                fig.suptitle(f"Binarizované spektrum | bs: {params[0]} | th: {params[1]} |")
    plt.show()