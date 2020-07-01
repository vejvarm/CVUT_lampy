from scipy import signal
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

from bin.flags import FLAGS


def load_accs(path, naccs=FLAGS.naccs):
    df = loadmat(path)

    accs = np.array([df[f'Acc{i}'] for i in range(1, naccs + 1)])  # [naccs, nsamples, nmeas]

    return accs


def make_filter(order, Wh=(0.45, 0.55)):
    num, denom, *_ = signal.butter(order, Wh, 'bandstop')
    return num, denom


def filter_data(num, denom, x):
    return signal.filtfilt(num, denom, x)

def plot_compare(x, y1, y2, legend=("y1", "y2"), xlabel="x", ylabel="y"):
    plt.semilogy(x, y1)
    plt.semilogy(x, y2)
    plt.legend(legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)



if __name__ == '__main__':
    path = "../data/trening/neporuseno/2months/08072018_AccM.mat"

    accs = load_accs(path)

    f_order = 5
    f_range = np.array((40, 60))  # Hz
    fs = 512  # Hz
    f_nyquist = fs/2  # Hz

    b, a = make_filter(f_order, f_range/f_nyquist)

    print(b.shape, a.shape)

    x_sample  = accs[0, :, 0]  # 1 acc time series
    time = np.arange(0, x_sample.shape[0]/fs, 1/fs)

    x_filtered = filter_data(b, a, x_sample)

    # compare in time domain
    plt.figure()
    plot_compare(time, x_sample, x_filtered)

    # calculate psd
    ns_per_hz = 10
    Nfft = fs*ns_per_hz  # 5120
    freqs = np.arange(0, fs, 1/ns_per_hz)
    X_sample = np.abs(np.fft.fft(x_sample, Nfft))**2/Nfft
    X_filtered = np.abs(np.fft.fft(x_filtered, Nfft))**2/Nfft

    plt.figure()
    plot_compare(freqs, X_sample, X_filtered)

    plt.show()
