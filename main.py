from collections import Counter

from flags import FLAGS
from preprocessing import Preprocessor
from Methods import M2

import numpy as np
from matplotlib import pyplot as plt


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
    setting = "training"
    folder = FLAGS.paths[setting]["folder"]
    dataset = FLAGS.paths[setting]["dataset"]
    period = [FLAGS.paths[setting]["period"]]*len(dataset)
    filename = ["X.npy"]*len(dataset)
    paths = [f"./{folder}/{d}/{p}/{f}" for d, p, f in zip(dataset, period, filename)]
    from_existing_file = True

    # multiscale params
    bin_sizes = (160, )
    thresholds = (.01, )
    plot_distributions = True

    # periodic params
    ndays = 60
    period = 1

    # define instance of Preprocessor and initialize M2
    preprocessor = Preprocessor()
    m2 = M2(preprocessor, from_existing_file=from_existing_file)

    # Train the method on 2 months of neporuseno (trained)
    m2.train(paths[0], bin_sizes, thresholds)

    # Calculate cross entropy of
    ce2 = m2.compare(paths[1], period=period, print_results=False)  # trained with neporuseno2
    ce3 = m2.compare(paths[2], period=period, print_results=False)  # trained with poruseno
    ce23 = np.vstack((ce2, ce3))
    # print(ce2.shape)  # (nperiods, nbins*nthresholds)

    # Find the highest cross-entropy for each day
    ce2_best_params, ce2_periodic_best = calc_periodic_best(ce2, bin_sizes, thresholds)
    ce3_best_params, ce3_periodic_best = calc_periodic_best(ce3, bin_sizes, thresholds)

    # Count which and how many times have combinations of (bin_size, threshold) been chosen as highest ce
    ce2_best_js = Counter(ce2_best_params)
    ce3_best_js = Counter(ce3_best_params)
    print(ce2_best_js)
    print(ce3_best_js)

    # Calculate cummulative sum of cross-entropies
    x = np.arange(0, len(ce2_periodic_best), 1)[:ndays]
    y2 = np.cumsum(ce2_periodic_best)[:ndays]
    y3 = np.cumsum(ce3_periodic_best)[:ndays]

    # Calculate params for linear regressions
    a2, b2 = linear_regression(y2, x)
    a3, b3 = linear_regression(y3, x)

    # plot the results of cummulative cross-entropies and their regression
    plt.plot(x, a2*x + b2, "b", label=f"regrese neporušených dat (a={a2:.1f}, b={b2:.1f})")
    plt.plot(x, a3*x + b3, "r", label=f"regrese porušených dat (a={a3:.1f}, b={b3:.1f})")
    plt.stem(y2, markerfmt="bx", linefmt="none", use_line_collection=True, label="neporušená lampa")
    plt.stem(y3, markerfmt="r+", linefmt="none", use_line_collection=True, label="porušená lampa")
    plt.xlabel(f"perioda ({period} " + ("den" if period == 1 else "dnů") + ")")
    plt.ylabel("kumulativní křížová entropie")
    plt.title("Porovnání kumulativních křížových entropií")
    plt.legend()

    # save the resulting plot
    plt.savefig(f"./images/M2/cummul_ce_nd-{ndays}_p-{period}.pdf")

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
                fig.suptitle(f"Binarizované spektrum {dataset[j]} | bs: {params[0]} | th: {params[1]} |")

    plt.show()