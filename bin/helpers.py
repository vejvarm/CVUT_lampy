import logging
import os

import numpy as np
from matplotlib import pyplot as plt


def plotter(x, y, ax=None, title="", xlabel="", ylabel="", label=""):
    if ax == None:
        fig, ax = plt.subplots()
    ax.plot(x, y, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_distributions_fn(distributions, save_path="../images/M2/binarized-spectra/"):
    save_path = os.path.normpath(save_path)
    os.makedirs(save_path, exist_ok=True)
    for j, dist in enumerate((distributions, )):
        for params, freqs, d in dist:
            nrows = 1
            ncols = 1
            fig, axes = plt.subplots(nrows, ncols)
            if nrows + ncols == 2:
                axes = np.array([axes])
            for i, ax in enumerate(axes.flatten()):
                y_pos = np.arange(len(d[i, :]))
                ax.bar(y_pos, d[i, :], align="center", width=0.9)
                ax.set_xlabel("bin")
                ax.set_ylabel("$psd_{bin}$")
                ax.set_yscale("log")
            fig.suptitle(f"Binarized spectrum | bin size: {params[0]} | threshold: {params[1]} |")
            flnm = f"{save_path}/binarized_spectrum_bs-{params[0]}_th-{params[1]}_"
            plt.savefig(f"{flnm}.png", dpi=200)
            plt.savefig(f"{flnm}.pdf")
            plt.savefig(f"{flnm}.svg")


def console_logger(name=__name__, level=logging.WARNING):

    if not isinstance(level, (str, int)):
        raise TypeError("Logging level data type is not recognised. Should be str or int.")

    logger = logging.getLogger(name)

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(level)
    formatter = logging.Formatter('%(levelname)7s (%(name)s) %(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger
