import numpy as np
from matplotlib import pyplot as plt

from bin.helpers import plotter
from bin.Generator import Generator
from bin.Preprocessor import Preprocessor

if __name__ == "__main__":
    savefigs = True

    nsamples = 15360  # počet časových vzorků signálu
    nsignals = 1 # počet signálů (počet měření za den)
    fs = 512  # vzorkovací frekvence
    nfft = 5120  # délka fourierovy transformace

    # parametry signálu
    fvs = np.array([5., 10., 30., 80., 110.])  # vlastní frekvence
    v_amp = (2., 5.)  # rozsah amplitud vlastních frekvencí
    v_shift = (-1., 1.)  # rozsah posunu vlastních frekvencí

    # parametry šumu
    fns = np.arange(512)  # šumové frekvence
    n_amp = (0, 2)
    n_shift = (0, 0)

    # vytvoření instancí generátorů
    g_sig = Generator(fs, fvs, v_amp, v_shift)  # pro signál
    g_noise = Generator(fs, fns, n_amp, n_shift)  # pro šum

    # vygenerování polí
    x = g_sig.generate(nsignals, nsamples)  # signálů
    n = g_noise.generate(nsignals, nsamples)  # šumů

    # normalizace
#    x = ((x.T - x.mean(axis=-1))/x.std(axis=-1)).T
#    n = ((n.T - n.mean(axis=-1))/n.std(axis=-1)).T

    # kombinace šumu a signálu
    xn = x + n
    xn = ((xn.T - xn.mean(axis=-1))/xn.std(axis=-1)).T
    print(xn.shape)

    # ověření frekvencí pomocí FFT
    X = abs(np.fft.fft(x, nfft))[:, :nfft//2]
    N = abs(np.fft.fft(n, nfft))[:, :nfft//2]
    XN = abs(np.fft.fft(xn, nfft))[:, :nfft//2]

    # Aplikace preprocessoru
    p = Preprocessor(use_autocorr=True, rem_neg=False)
    freq_vals, psd = p.simple_preprocess(xn.T)

    # Vykreslení výsledků
    t = np.arange(nsamples)/fs
    # výstupy generátorů:
    fig, ax = plt.subplots(3, 1)
    plotter(t, x.T, ax[0], title="signal")
    plotter(t, n.T, ax[1], title="noise")
    plotter(t, xn.T, ax[2], title="signal + noise", xlabel="time (s)")
    if savefigs:
        plt.tight_layout()
        plt.savefig("../data/generated/example/01_signals.pdf")
        plt.savefig("../data/generated/example/01_signals.svg")

    # FFT signálů
    fss = np.arange(nfft//2)/nfft*fs
    fig, ax = plt.subplots(3, 1)
    plotter(fss, X.T, ax[0], title="FFT of signal", ylabel="abs(fft)")
    plotter(fss, N.T, ax[1], title="FFT of noise", ylabel="abs(fft)")
    plotter(fss, XN.T, ax[2], title="FFT of signal + noise", xlabel="Frequency (Hz)", ylabel="abs(fft)")
    if savefigs:
        plt.tight_layout()
        plt.savefig("../data/generated/example/02_ffts.pdf")
        plt.savefig("../data/generated/example/02_ffts.svg")

    # Preprocessing zašumělých signálů a jejich průměr
    plotter(freq_vals, psd.mean(axis=-1), title="Mean preprocessed spectrum", xlabel="Frequency (Hz)", ylabel="psd")
    if savefigs:
        plt.tight_layout()
        plt.savefig("../data/generated/example/03_psd.pdf")
        plt.savefig("../data/generated/example/03_psd.svg")
    plt.show()