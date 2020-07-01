import numpy as np

from matplotlib import pyplot as plt
from matplotlib import animation as animation


def _animate_init(naccs, freqs, PSD_means, fig):
    ax = list()
    line = list()
    grid = plt.GridSpec(naccs//2, 2, wspace=0.4, hspace=0.3)

    def inner():
        # acc plots (first 3 rows of figure)
        for i in range(naccs):
            ax.append(fig.add_subplot(grid[i // 2, i % 2]))
            line.append(ax[i].semilogy(freqs, np.ones(freqs.shape)*PSD_means[i])[0])
            line[i].set_ydata([np.nan] * len(freqs))
            line[i].set_color("red")
            ax[i].set_ylim(0, 1e1)
        return line

    return inner, line


def animate(j, psd_array, line):
    """

    :param j:
    :param psd_array: (naccs, nfft/2, nfiles*nmeas -- this will be animated)
    :param line:
    :return: line
    """
    for i in range(naccs):
        psd = psd_array[i, :, j]
        if any(np.isnan(psd)):
            pass
        else:
            line[i].set_ydata(psd)
    return line


if __name__ == '__main__':
    root = "D:/!private/Lord/Git/CVUT_lampy"
    datasets = ["neporuseno"]  # , "neporuseno2", "poruseno"
    period = "2months"  # week or 2months
    paths = [f"{root}/data/{d}/{period}" for d in datasets]

    for path in paths:
        try:
            freqs = np.load(path+"/freqs.npy")
            PSD = np.load(path+"/PSD.npy")
            psd_array_excited = np.load(path+"/psd_array_excited.npy")
            psd_array_unexcited = np.load(path + "/psd_array_unexcited.npy")
        except FileNotFoundError:
            raise FileNotFoundError("Categorized array files or 'freqs' file not found in given path,"
                                    " have you ran categorization.py before?")

        nfiles, naccs, nfft, nmeas = psd_array_excited.shape

        # calculate mean of PSDs
        PSD_means = PSD.mean(axis=1)

        # reshape to (naccs, nfft/2, nfiles*nmeas)
        psd_excited = psd_array_excited.transpose(1, 2, 0, 3).reshape(naccs, nfft, nfiles * nmeas)
        psd_unexcited = psd_array_unexcited.transpose(1, 2, 0, 3).reshape(naccs, nfft, nfiles * nmeas)

        fig = plt.figure()
        animate_init, line = _animate_init(naccs, freqs, PSD_means, fig)
        ani1 = animation.FuncAnimation(fig, animate, init_func=animate_init, fargs=(psd_excited, line),
                                       frames=naccs * nmeas,
                                       interval=200,
                                       blit=True, save_count=50, repeat=True)

        fig = plt.figure()
        animate_init, line = _animate_init(naccs, freqs, PSD_means, fig)
        ani2 = animation.FuncAnimation(fig, animate, init_func=animate_init, fargs=(psd_unexcited, line),
                                       frames=naccs * nmeas,
                                       interval=200,
                                       blit=True, save_count=50, repeat=True)

        plt.show()