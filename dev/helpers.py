import logging

from matplotlib import pyplot as plt


def plotter(x, y, ax=None, title="", xlabel="", ylabel="", label=""):
    if ax == None:
        fig, ax = plt.subplots()
    ax.plot(x, y, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def console_logger(name="__main__", level=logging.WARNING):

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
