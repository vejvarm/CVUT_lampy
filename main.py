from matplotlib import pyplot as plt

from preprocessing import mat2dict, calc_zscore, calc_psd, remove_noise

if __name__ == '__main__':
    df = mat2dict('./data/neporuseno/03092018_AccM.mat')
    print(df.keys())
    fs = 512             # acquisition sampling frequency
    ns_per_hz = 10       # desired number of samples per Hertz in FFT

    acc1 = df['Acc1']

    acc1_zscore = calc_zscore(acc1)
    freq_vals, acc1_psd = calc_psd(acc1_zscore, fs, ns_per_hz)

    acc1_psd_denoised = remove_noise(acc1_psd, fs)

    plt.plot(freq_vals, acc1_psd)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('PSD (Power Spectral Density)')
    plt.show()


