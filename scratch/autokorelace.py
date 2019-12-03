# výpočet autokorelační funkce (ACF)
# metodika viz http://dsp.vscht.cz/konference_matlab/MATLAB09/prispevky/079_pavlik.pdf
import numpy as np

from scipy.linalg import hankel
from matplotlib import pyplot as plt

N = 15360
fs = 512
facc = 64


def gensig(N, fs, facc):
    t = np.arange(N)/fs
    amp = abs(np.sin(2*np.pi*t/10))
    signal = amp*np.sin(2*np.pi*facc*t)
    noise = np.random.normal(0, 3.0, (N, ))
    return t, signal, signal+noise


def autocorr(x):
    """ autokorelační funkce """
    N = x.shape[0]  # počet vzorků signálu
    XX = hankel(x[1:])  # vytvoření hankelovy matice z prvků x[1] až x[N-1] (horní levá trojúhelníková matice)
    vX = x[:N-1]  # vektor x[0] až x[N-2]
    Rrr = np.matmul(XX, vX)/N - x.mean()**2  # výpočet normalizované ACF
    return Rrr

def abs_fft(x, N):
   xfft = np.fft.fft(x, N)
   return np.abs(xfft)**2/N

def main():
    time, signal, noisy_signal = gensig(N, fs, facc)
    Rrr = autocorr(noisy_signal)

    print(Rrr.shape)

    freqs = np.arange(N)/N*fs
    sigfft = abs_fft(signal, N)
    nsigfft = abs_fft(noisy_signal, N)
    Rrrfft = abs_fft(Rrr, N)

    print(Rrrfft.shape)

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(time, signal)
    ax[1].plot(time, noisy_signal)
    ax[2].plot(Rrr)

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(freqs, sigfft)
    ax[1].plot(freqs, nsigfft)
    ax[2].plot(freqs, Rrrfft)

    plt.show()

if __name__ == '__main__':
    main()