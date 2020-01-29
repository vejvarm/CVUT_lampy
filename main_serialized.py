from collections import Counter

from flags import FLAGS
from preprocessing import Preprocessor
from Methods import M2

import numpy as np
from matplotlib import pyplot as plt

# TODO: korelace mezi sílou větru, rychlostí větru a amplitudou (vybuzeností) vlastních frekvencí

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
    # Paths to training files
    setting = "training"
    folder = FLAGS.paths[setting]["folder"]
    dataset = FLAGS.paths[setting]["dataset"]
    period = [FLAGS.paths[setting]["period"]]*len(dataset)
    filename = ["X_l2.npy"]*len(dataset)
    paths = [f"./{folder}/{d}/{p}/{f}" for d, p, f in zip(dataset, period, filename)]

    # Paths to validation files
    setting_valid = "serialized"
    folder = FLAGS.paths[setting_valid]["folder"]
    dataset = FLAGS.paths[setting_valid]["dataset"]
    period = [FLAGS.paths[setting_valid]["period"]]*len(dataset)
    filename = ["X_l2.npy"] * len(dataset)
    paths_valid = [f"./{folder}/{d}/{p}/{f}" for d, p, f in zip(dataset, period, filename)]

    from_existing_file = True

    # multiscale params
    bin_sizes = (10, 20, 40, 80,)
    thresholds = (0.1, 1., 10.)
    plot_distributions = False

    # periodic params
    period = 1
    ndays_unbroken = FLAGS.serialized["unbroken"]//period
    ndays_broken = FLAGS.serialized["broken"]//period
    ndays = ndays_unbroken + ndays_broken


    # define instance of Preprocessor and initialize M2
    preprocessor = Preprocessor()
    m2 = M2(preprocessor, from_existing_file=from_existing_file)

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
    plt.savefig(f"./images/M2/cummul_ce_nd-{ndays}_p-{period}.pdf")
    plt.show()