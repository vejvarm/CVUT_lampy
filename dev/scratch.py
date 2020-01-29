def generate_fft(fs, nfft, fvs, amp, shift, nsamples):
    ns_per_hz = nfft//fs
    fv_fft = (fvs+shift).astype(np.int32)*ns_per_hz

    X = np.zeros((nfft//2, ), dtype=np.float32)

    X[fv_fft] = amp
    X = np.hstack([X, X[::-1]])

    plt.figure()
    plt.plot(X)

    # transformace do časové oblasti
    x = np.fft.ifft(X, nsamples)

    return x


def sine_wave(i, fs, fv, amp):
    return amp*np.sin(2*np.pi*fv*i/fs)


def generate(i, fs, fvs, amp_range, shift_range):
    amp = np.random.uniform(*amp_range, (len(fvs), i.shape[0], 1))
    shift = np.random.uniform(*shift_range, (len(fvs), i.shape[0], 1))
    return np.array([sine_wave(i, fs, fv+sft, amp) for fv, amp, sft in zip(fvs, amp, shift)]).sum(axis=0)