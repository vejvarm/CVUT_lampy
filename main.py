# short/long time averaging
# TODO: plotting
import numpy as np

from matplotlib import pyplot as plt

from preprocessing import Preprocessor
from Aggregators import Average, complete_average


if __name__ == '__main__':
    path_list = ['./data/neporuseno/week']
    # path_list = ['./data/ruzne/PovlakPrycSloup1OdleptaniSloup2Sloup3__2019_03_28___2019_07_01__vz/week']
    averages_dict = dict()
    # Preprocess files referenced by path_list
    preproc = Preprocessor(noise_f_rem=(2, 50, 100, 150, 200),
                           noise_df_rem=(2, 5, 1, 5, 1),
                           mov_filt_size=5)  # refer to __init__ for possible preprocessing settings
    preprocessed = preproc.run(path_list)
#    print(preprocessed.keys())
#    print(preprocessed['12102018_AccM'])

    num_accs = len(preprocessed[list(preprocessed.keys())[0]][1])
    long_term_aggs = [Average() for _ in range(num_accs)]
    for key, value in preprocessed.items():
        freq_vals, psd_list, _, _ = value
        daily_avg = [[]] * len(psd_list)
        for i, psd in enumerate(psd_list):
            daily_avg[i] = complete_average(psd)    # in production use Aggregators.Average class for online averaging
            long_term_aggs[i].update(daily_avg[i])  # aggregate the long term average (PSD) of the psd of accelerations

        averages_dict[key] = daily_avg  # save calculated daily psd averages of acc[0:6]

#    plt.plot(freq_vals, acc1_psd_denoised[:, 0])
    _, ax = plt.subplots(3, 2)
    ax = ax.flatten()
    for i, agg in enumerate(long_term_aggs):
        ax[i].plot(freq_vals, agg.PSD)
        ax[i].set_xlabel('frequency [Hz]')
        ax[i].set_ylabel('PSD')
        ax[i].set_title(f'sloup {i//2 + 1} acc {i%2 + 1}')

#    for value in averages_dict.values():
#        plt.plot(freq_vals, value[0])

    plt.show()


