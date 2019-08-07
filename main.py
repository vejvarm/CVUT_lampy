# TODO: plotting and short/long time averaging
from matplotlib import pyplot as plt

from preprocessing import Preprocessor, complete_average
from Aggregators import Average

if __name__ == '__main__':
    path_list = ['./data/poruseno/']
    # Preprocess files referenced by path_list
    preproc1 = Preprocessor(noise_f_rem=(50, 100, 150, 200),
                            noise_df_rem=(20, 1, 10, 1))  # refer to __init__ for possible preprocessing settings
    preprocessed = preproc1.run(path_list)
    print(preprocessed.keys())
    freq_vals, psd_list, _, _ = preprocessed[list(preprocessed.keys())[0]]
    acc1_psd = psd_list[0]
    acc1_psd_avg = complete_average(acc1_psd)

    # using Aggregator
    short_term_avg = Average(acc1_psd[:, 0])
    for i in range(1, acc1_psd.shape[1]):
        short_term_avg.update(acc1_psd[:, i])

#    plt.plot(freq_vals, acc1_psd_denoised[:, 0])
    plt.plot(freq_vals, acc1_psd_avg)
    plt.plot(freq_vals, short_term_avg.PSD)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('PSD (Power Spectral Density)')
    plt.show()


