import numpy as np


class Generator:

    def __init__(self, fs=512, fvs=(5., 10., 30., 80., 110.), amp_range=(1., 5.), shift_range=(0., 0.)):
        """ Generátor signálů s několika vlastními frekvencemi, náhodnými amplitudami a frekvenčním posunutím

        :param fs: (int) vzorkovací frekvence
        :param fvs: (int) vlastní frekvence generovaného signálů
        :param amp_range: (Seq[int, int]) rozsah amplitud vlastních frekvencí
        :param shift_range: (Seq[int, int]) rozsah frekvenčního posunu vlastních frekvencí
        """

        self.fs = fs
        self.fvs = np.array(fvs)
        self.amp_range = amp_range
        self.shift_range = shift_range

        self.nfvs = len(self.fvs)

    @staticmethod
    def _sine_wave(i, fs, fv, amp):
        return amp * np.sin(2 * np.pi * fv * i / fs)

    def generate(self, nsig, nsamp):
        """ generátor výsledného signálu dle parametrů v inicializaci třídy

        :param nsig: (int) počet signálů
        :param nsamp: (int) počet vzorků v signálu

        :return signals: (2dARRAY[nsignals, nsamples]) vygenerované signály dle zadaných parametrů
        """
        grid = np.vstack([np.arange(nsamp)]*nsig)
        amp = np.random.uniform(*self.amp_range, (self.nfvs, nsig, 1))
        shift = np.random.uniform(*self.shift_range, (self.nfvs, nsig, 1))
        return np.array([self._sine_wave(grid, self.fs, fv + sft, amp)
                         for fv, amp, sft in zip(self.fvs, amp, shift)]).sum(axis=0)