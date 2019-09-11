import numpy as np

from matplotlib import pyplot as plt
from matplotlib import animation as animation

from preprocessing import Preprocessor

# Loading data
path_list = ['D:/!private/Lord/Git/CVUT_lampy/data/neporuseno/week']
preproc = Preprocessor(noise_f_rem=(2, 50, 100, 150, 200),
                           noise_df_rem=(2, 5, 1, 5, 1),
                           mov_filt_size=5)  # refer to __init__ for possible preprocessing settings
preprocessed = preproc.run(path_list)

freqs, accs, wind_dir, wind_spd = preprocessed['09092018_AccM']
wind_spd = wind_spd.flatten()

# Animating data
fig = plt.figure()

grid = plt.GridSpec(4, 2, wspace=0.4, hspace=0.3)

n_accs = len(accs)
n_msrmnts = len(wind_spd)

ax = list()
line = list()

# acc plots (first 3 rows of figure)
for i in range(n_accs):
    ax.append(fig.add_subplot(grid[i//2, i%2]))
    line.append(ax[i].plot(freqs, np.zeros(freqs.shape))[0])

# wind speed plot (4th row of figure)
ax.append(fig.add_subplot(grid[3, :]))
line.append(ax[-1].plot(np.arange(n_msrmnts), np.zeros(n_msrmnts))[0])

def animate_init():
    for i in range(n_accs):
        line[i].set_ydata([np.nan]*len(freqs))
        ax[i].set_ylim(0, np.max(accs[i]))
    line[-1].set_ydata([np.nan]*n_msrmnts)
    ax[-1].set_ylim(0, np.max(wind_spd))
    return line

def animate(j):
    for i in range(n_accs):
        line[i].set_ydata(accs[i][:, j])
    line[-1].set_ydata(np.hstack((wind_spd[:j], [np.nan]*(n_msrmnts - j))))
    return line

if __name__ == '__main__':
    ani = animation.FuncAnimation(fig, animate, init_func=animate_init, interval=100, blit=True, save_count=50, repeat=True)

    plt.show()