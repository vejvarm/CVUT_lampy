import numpy as np
import SimpSOM as sps

from preprocessing import Preprocessor

if __name__ == '__main__':
    # Loading data
    path_list = ['D:/!private/Lord/Git/_CVUT_lampy/data/neporuseno/week']
    som_weights = None
    preproc = Preprocessor(ns_per_hz=1,
                           noise_f_rem=(2, 50, 100, 150, 200),
                           noise_df_rem=(2, 5, 1, 5, 1),
                           mov_filt_size=5)  # refer to __init__ for possible preprocessing settings
    preprocessed = preproc.run(path_list)

    accs = list()

    for key, (freqs, acc, wind_dir, wind_spd) in preprocessed.items():
        acc_stacked = np.hstack(acc)
        accs.append(acc_stacked)

    data = np.hstack(accs)

    print(data.shape)

    # need to transpose (rows != samples, columns != features)
    data = data.T

    # SOM Network
    # Building net
    net = sps.somNet(2, 2, data, loadFile=som_weights if som_weights else None, PBC=True)

    if not som_weights:
        # Training net
        net.train(0.1, 100000)
        # Save weights to file
        net.save('som_weights')

    # Print a map of the network nodes and colour them according to the first feature (column number 0) of the dataset
    # and then according to the distance between each node and its neighbours.
    # net.nodes_graph(colnum=80)
    # net.diff_graph()

    test_data = data[0:100, :]
    test_labels = np.arange(0, len(test_data))

    # Project the datapoints on the new 2D network map.
    net.project(data[0:100, :], labels=test_labels)

    # Cluster the datapoints according to the Quality Threshold algorithm.
    net.cluster(data[0:100, :], type='qthresh')