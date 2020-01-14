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
    x = np.fft.ifftshift((x - x.mean()) / x.std())
    XX = hankel(x[1:])  # vytvoření hankelovy matice z prvků x[1] až x[N-1] (horní levá trojúhelníková matice)
    vX = x[:N-1]  # vektor x[0] až x[N-2]
    Rrr = np.matmul(XX, vX)/N - x.mean()**2  # výpočet normalizované ACF
    return Rrr

def fft_autocorr(x):
    """ autokorelace ve frekvenční oblasti """
    # přirozená a nejrychlejší metoda!
    N = x.shape[0]
    x = np.fft.ifftshift((x - x.mean()) / x.std())
    x = np.pad(x, (N//2, N//2), mode="constant")
    X = np.fft.fft(x)
    XX = np.abs(X)**2
    Rxx = np.fft.ifft(XX)[:N:-1]
    return np.real(Rxx)/N - x.mean()**2

def np_autocorr(x):
    """ autokorelace pomocí numpy """
    N = x.shape[0]
    x = np.fft.ifftshift((x - x.mean()) / x.std())
    Rxx = np.correlate(x, x, mode="full")/N - x.mean()**2
    return Rxx[N:]

def abs_fft(x, N):
   xfft = np.fft.fft(x, N)
   return np.abs(xfft)**2/N

def main():
    time, signal, noisy_signal = gensig(N, fs, facc)
    Rrr = autocorr(noisy_signal)
    Rxxf = fft_autocorr(noisy_signal)
    Rxxnp = np_autocorr(noisy_signal)

    print(Rrr.shape)

    freqs = np.arange(N)/N*fs
    sigfft = abs_fft(signal, N)
    nsigfft = abs_fft(noisy_signal, N)
    Rrrfft = abs_fft(Rrr, N)
    Rxxffft = abs_fft(Rxxf, N)
    Rxxnpfft = abs_fft(Rxxnp, N)

    print(Rrrfft.shape)

    fig, ax = plt.subplots(5, 1)
    ax[0].plot(time, signal)
    ax[1].plot(time, noisy_signal)
    ax[2].plot(Rrr)
    ax[3].plot(Rxxf)
    ax[4].plot(Rxxnp)

    fig, ax = plt.subplots(5, 1)
    ax[0].plot(freqs, sigfft)
    ax[1].plot(freqs, nsigfft)
    ax[2].plot(freqs, Rrrfft)
    ax[3].plot(freqs, Rxxffft)
    ax[4].plot(freqs, Rxxnpfft)

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(time[:-1], Rrr)
    ax[0].set_title("Původní implementace (časová doména maticově)")
    ax[1].plot(time[:-1], Rxxf)
    ax[1].set_title("Nová implementace (frekvenční doména)")
    ax[2].plot(time[:-1], Rrr - Rxxf)
    ax[2].set_title("Rozdíl")
    ax[2].set_xlabel("čas")
    ax[0].set_ylabel("Rxx_hank")
    ax[1].set_ylabel("Rxx_fft")
    ax[2].set_ylabel("Rxx_hank - Rxx_fft")
    ax[2].set_ylim((-0.04, 0.04))
    plt.suptitle("Autokorelační funkce")

    plt.show()

if __name__ == '__main__':
    main()