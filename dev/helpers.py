from matplotlib import pyplot as plt


def plotter(x, y, ax=None, title="", xlabel="", ylabel="", label=""):
    if ax == None:
        fig, ax = plt.subplots()
    ax.plot(x, y, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)