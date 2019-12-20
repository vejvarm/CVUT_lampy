import numpy as np

from matplotlib import pyplot as plt
from matplotlib import animation as animation

from preprocessing import Preprocessor

# Loading data
path_list = ['../data/neporuseno/week']
preproc = Preprocessor(noise_f_rem=(2, 50, 100, 150, 200),
                           noise_df_rem=(2, 5, 1, 5, 1),
                           mov_filt_size=5)  # refer to __init__ for possible preprocessing settings
preprocessed = preproc.run(path_list)

freqs, accs, wind_dir, wind_spd = preprocessed[list(preprocessed.keys())[0]]
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
    ax.append(fig.add_subplot(grid[i//2 + 1, i % 2]))
    line.append(ax[i].semilogy(freqs, np.ones(freqs.shape))[0])
    for j in range(n_msrmnts):
        acc = accs[i][:, j]
        acc_min, acc_max = (np.min(acc), np.max(acc))
#        mean = np.mean(accs[i][:, j])
#        std = np.std(accs[i][:, j])
        acc = (acc - acc_min)/(acc_max - acc_min)

# wind speed plot (4th row of figure)
idx = np.arange(n_msrmnts)
ax.append(fig.add_subplot(grid[0, :]))
line.append(ax[-1].plot(idx, np.zeros(n_msrmnts))[0])  # 1 for consistent line
line.append(ax[-1].plot(idx, np.zeros(n_msrmnts))[0])  # 1 for current position

def animate_init():
    for i in range(n_accs):
        line[i].set_ydata([np.nan]*len(freqs))
        line[i].set_color("red")
        ax[i].set_ylim(np.min(accs[i]), np.max(accs[i]))
    line[-1].set_ydata([np.nan]*n_msrmnts)
    line[-1].set_color("red")
    line[-1].set_linewidth(2)
    line[-2].set_ydata(wind_spd)
    ax[-1].set_ylim(-0.2, np.max(wind_spd))
    return line

def animate(j, ):
    for i in range(n_accs):
        line[i].set_ydata(accs[i][:, j])
#    line[-1].set_ydata(np.hstack((wind_spd[:j], [np.nan]*(n_msrmnts - j))))
    line[-1].set_data(idx[j-5:j], wind_spd[j-5:j])
#    ax[-1].axvspan(idx[j], idx[j+3], color="red")
    return line

if __name__ == '__main__':
    ani = animation.FuncAnimation(fig, animate, init_func=animate_init, fargs=None, frames=n_msrmnts, interval=200,
                                  blit=True, save_count=50, repeat=True)

    plt.show()