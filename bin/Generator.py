import numpy as np


class Generator:

    def __init__(self, fs=512, fvs=(5., 10., 30., 80., 110.), amp_range=(1., 5.), shift_range=(0., 0.)):
        """ Generátor signálů s několika vlastními frekvencemi, náhodnými amplitudami a frekvenčním posunutím

        :param fs: (int) vzorkovací frekvence
        :param fvs: (Seq[int]) vlastní frekvence generovaného signálů
        :param amp_range: (Seq[int, int]) rozsah amplitud vlastních frekvencí
        :param shift_range: (Seq[int, int]) rozsah frekvenčního posunu vlastních frekvencí
        """

        self.fs = fs
        self.fvs = np.array(fvs)
        self.amp_range = amp_range
        self.shift_range = shift_range

        self.nfvs = len(self.fvs)

        self.amp = None
        self.shift = None

    @staticmethod
    def _sine_wave(i, fs, fv, amp):
        return amp * np.sin(2 * np.pi * fv * i / fs)

    def generate(self, nsig, nsamp, omit=tuple()):
        """ generátor výsledného signálu dle parametrů v inicializaci třídy

        :param nsig: (int) počet signálů
        :param nsamp: (int) počet vzorků v signálu
        :param omit: (Seq[int]) vynechání vlastních frekvencí v signálech v pořadí dle indexů v omit.
                     délka omit musí být stejná jako nsig!
                     -1 znamená, že nemá být vynechána žádná frekvence
                     např. omit=(0, -1, 1) vynechá v 0. signálu 0. vl. frekvenci, a v 2. signálu 1. vl. frekvenci

        :return signals: (2dARRAY[nsignals, nsamples]) vygenerované signály dle zadaných parametrů
        """
        grid = np.vstack([np.arange(nsamp)]*nsig)
        amp = np.random.uniform(*self.amp_range, (self.nfvs, nsig, 1))
        shift = np.random.uniform(*self.shift_range, (self.nfvs, nsig, 1))
        if not omit:
            return np.array([self._sine_wave(grid, self.fs, fv + sft, amp)
                             for fv, amp, sft in zip(self.fvs, amp, shift)]).sum(axis=0)
        else:
            assert len(omit) == nsig, "Length of 'omit' sequence must be the same as nsig!"
            assert max(omit) == self.nfvs - 1, "The highest value in 'omit' must not be higher than len(fvs) - 1!"
            waves = []
            for i, (fv, am, sft) in enumerate(zip(self.fvs, amp, shift)):
                sigs = []
                for g, a, s, o in zip(grid, am, sft, omit):
                    if o == i:
                        a = 0  # omit this signal
                    sigs.append(self._sine_wave(g, self.fs, fv + s, a))
                sigs_np = np.vstack(sigs)
                waves.append(sigs_np)
            return np.array(waves).sum(axis=0)