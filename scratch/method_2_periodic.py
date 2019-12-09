from collections import Counter

from flags import FLAGS
from preprocessing import Preprocessor
from Methods import M2

import numpy as np
from matplotlib import pyplot as plt

# TODO: bin statistics

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
    setting = "training"
    folder = FLAGS.paths[setting]["folder"]
    dataset = FLAGS.paths[setting]["dataset"]
    period = [FLAGS.paths[setting]["period"]]*len(dataset)
    filename = ["X.npy"]*len(dataset)
    paths = [f"../{folder}/{d}/{p}/{f}" for d, p, f in zip(dataset, period, filename)]
#    paths[1] = f"../data/validace/neporuseno/X.npy"
#    paths[2] = f"../data/validace/poruseno/X.npy"

    print(paths)

    preprocessor = Preprocessor()

    from_existing_file = True
    var_scaled_PSD = False

    m2 = M2(preprocessor, var_scaled_PSD=var_scaled_PSD)

    # multiscale params
    bin_sizes = (80, )
    thresholds = (0.1, 0.2, 0.5)
    plot_distributions = False

    # Train the method on 2 months of neporuseno
    m2.train(paths[0], bin_sizes, thresholds)

    ce2 = m2.compare(paths[1], period=1, print_results=False)
    ce3 = m2.compare(paths[2], period=1, print_results=False)
    ce23 = np.vstack((ce2, ce3))

    # NORMALIZACE podle ce23???
#    ce2 = (ce2 - ce23.mean())/ce23.std()
#    ce3 = (ce3 - ce23.mean())/ce23.std()



    print(ce23.shape)

    ce2_diff = np.diff(ce2, axis=0)
    ce3_diff = np.diff(ce3, axis=0)
    ce23_diff = np.diff(ce23, axis=0)
    # print(ce2.shape)  # (nperiods, nbins*nthresholds)

    ce2_best_params, ce2_periodic_best = calc_periodic_best(ce2, bin_sizes, thresholds)  # ce2_diff lepší pro valid data
    ce3_best_params, ce3_periodic_best = calc_periodic_best(ce3, bin_sizes, thresholds)  # ce3_diff lepší pro valid data

    ce2_best_js = Counter(ce2_best_params)
    ce3_best_js = Counter(ce3_best_params)
    print(ce2_best_js)
    print(ce3_best_js)

    ndays = 60
    x = np.arange(0, len(ce2_periodic_best), 1)[:ndays]
    y2 = np.cumsum(ce2_periodic_best)[:ndays]
    y3 = np.cumsum(ce3_periodic_best)[:ndays]

    a2, b2 = linear_regression(y2, x)
    a3, b3 = linear_regression(y3, x)

    plt.plot(x, a2*x + b2, "b", label=f"regrese neporušených dat (a={a2:.1f}, b={b2:.1f})")
    plt.plot(x, a3*x + b3, "r", label=f"regrese porušených dat (a={a3:.1f}, b={b3:.1f})")
    plt.stem(y2, markerfmt="bx", linefmt="none", use_line_collection=True, label="neporušená lampa")
    plt.stem(y3, markerfmt="r+", linefmt="none", use_line_collection=True, label="porušená lampa")
    plt.xlabel("perioda (den)")
    plt.ylabel("kumulativní křížová entropie")
    plt.title("Porovnání kumulativních křížových entropií")
    plt.legend()
    plt.show()