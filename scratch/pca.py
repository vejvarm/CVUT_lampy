import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt

from preprocessing import Preprocessor


def load_and_standardize(path_list):

    # Load and preprocess data
    preproc = Preprocessor(ns_per_hz=1,
                           freq_range=(0, 256),
                           noise_f_rem=(2, 50, 100, 150, 200),
                           noise_df_rem=(2, 5, 1, 5, 1),
                           mov_filt_size=5)  # refer to __init__ for possible preprocessing settings
    preprocessed = preproc.run(path_list)

    accs = list()

    # Stack data into one numpy structure
    for key, (freqs, acc, wind_dir, wind_spd) in preprocessed.items():
        acc_stacked = np.hstack(acc[2:4])
        accs.append(acc_stacked)

    data = np.hstack(accs)

    # transpose to (samples, features)
    data = data.T

    # standardize individual features (frequencies)
    sc = StandardScaler()

#    return sc.fit_transform(data)
    return (data - data.mean())/data.std()


def plot_explained_variance(data):
    # PCA
    pca = PCA(n_components=None)
    pca.fit(data)

    pca_exp_var = pca.explained_variance_ratio_
    cum_pca_exp_var = np.cumsum(pca.explained_variance_ratio_)

    plt.figure()
    plt.bar(range(1, data.shape[1] + 1), pca_exp_var, alpha=0.5, align='center', label='vyjadřující variance')
    plt.step(range(1, data.shape[1] + 1), cum_pca_exp_var, where='mid', label='kumulativní vyjadřující variance')
    plt.xlabel('PCA index')
    plt.ylabel('míra vyjadřující variance')
    plt.legend()


def reduce_to_2D(data):
    pca = PCA(n_components=2)
    data_2D = pca.fit_transform(data)

    return data_2D, pca


if __name__ == '__main__':
    path_list = None
    data_new = load_and_standardize(['../data/neporuseno/2months'])
    data_old = load_and_standardize(['../data/poruseno/2months'])

    plot_explained_variance(data_new)
    plot_explained_variance(data_old)
    plt.show()

    data_2D_new, pca = reduce_to_2D(data_new)

    data_2D_old = pca.transform(data_old)

    plt.figure()
    plt.scatter(data_2D_new[:, 0], data_2D_new[:, 1], marker='o', edgecolors='blue')
    plt.scatter(data_2D_old[:, 0], data_2D_old[:, 1], marker='x', edgecolors='red')
    plt.show()

# TODO: každé zrychlení zvlášť?
