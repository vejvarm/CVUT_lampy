import numpy as np
from matplotlib import pyplot as plt

from dev.helpers import plotter
from preprocessing import Preprocessor


class Generator:

    def __init__(self, nsig, nsamp, fs, fvs, amp_range, shift_range):
        """ Generátor signálů s několika vlastními frekvencemi s náhodnými amplitudami a frekvenčním posunutím

        :param nsig: počet signálů
        :param nsamp: počet vzorků v signálu
        :param fs: vzorkovací frekvence
        :param fvs: vlastní frekvence generovaného signálů
        :param amp_range: rozsah amplitud vlastních frekvencí
        :param shift_range: rozsah frekvenčního posunu vlastních frekvencí
        """

        self.nsig = nsig
        self.nsamp = nsamp
        self.fs = fs
        self.fvs = fvs
        self.amp_range = amp_range
        self.shift_range = shift_range

        self.grid = np.vstack([np.arange(nsamp)]*nsig)
        self.nfvs = len(self.fvs)

    @staticmethod
    def _sine_wave(i, fs, fv, amp):
        """ generátor siusovek """
        return amp * np.sin(2 * np.pi * fv * i / fs)

    def generate(self):
        """ generátor výsledného signálu dle parametrů v inicializaci třídy

        :return signals: (nsignals, nsamples) 2D pole vygenerovaných signálů dle zadaných parametrů
        """
        amp = np.random.uniform(*self.amp_range, (self.nfvs, self.nsig, 1))
        shift = np.random.uniform(*self.shift_range, (self.nfvs, self.nsig, 1))
        return np.array([self._sine_wave(self.grid, self.fs, fv + sft, amp)
                         for fv, amp, sft in zip(self.fvs, amp, shift)]).sum(axis=0)


if __name__ == "__main__":
    nsamples = 15360  # počet časových vzorků signálu
    nsignals = 2  # počet signálů (počet měření za den)
    fs = 512  # vzorkovací frekvence
    nfft = 5120  # délka fourierovy transformace

    # parametry signálu
    fvs = np.array([5., 10., 30., 80., 110.])  # vlastní frekvence
    v_amp = (1, 5)  # rozsah amplitud vlastních frekvencí
    v_shift = (-4, 4)  # rozsah posunu vlastních frekvencí

    # parametry šumu
    fns = np.arange(250)  # šumové frekvence
    n_amp = (0, 1)
    n_shift = (0, 0)

    # vytvoření instancí generátorů
    g_sig = Generator(nsignals, nsamples, fs, fvs, v_amp, v_shift)  # pro signál
    g_noise = Generator(nsignals, nsamples, fs, fns, n_amp, n_shift)  # pro šum

    # vygenerování polí
    x = g_sig.generate()  # signálů
    n = g_noise.generate()  # šumů

    # kombinace šumu a signálu
    xn = x + n

    # ověření frekvencí pomocí FFT
    X = abs(np.fft.fft(xn, nfft))[:, :nfft//2]

    # Aplikace preprocessoru
    p = Preprocessor(use_autocorr=True, rem_neg=False)
    freq_vals, psd = p.simple_preprocess(xn.T)

    # Vykreslení výsledků
    t = np.arange(nsamples)/fs
    # výstupy generátorů:
    fig, ax = plt.subplots(3, 1)
    plotter(t, x.T, ax[0], title="signál")
    plotter(t, n.T, ax[1], title="šum")
    plotter(t, xn.T, ax[2], title="signál + šum", xlabel="čas (s)")

    # FFT zašumělých signálů
    fs = np.arange(nfft//2)/nfft*fs
    plotter(fs, X.T, title="Four. obrazy signálu", xlabel="Frekvence (Hz)", ylabel="abs(fft)")

    # Preprocessing zašumělých signálů a jejich průměr
    plotter(freq_vals, psd.mean(axis=-1), title="Předzpracovaný signál", xlabel="Frekvence (Hz)", ylabel="psd")
#    plt.plot(freq_vals, psd.mean(axis=-1))
    plt.show()