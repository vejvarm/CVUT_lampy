from flags import FLAGS

import numpy as np
from matplotlib import pyplot as plt


def abs_diff(x, y):
    return np.abs(x - y)


def normalize(x):
    return (x - x.min())/(x.max() - x.min())


if __name__ == '__main__':
    settings = ["training", "serialized"]
    for setting in settings:
        root = ".."
        folder = FLAGS.paths[setting]["folder"]
        dataset = FLAGS.paths[setting]["dataset"]
        period = [FLAGS.paths[setting]["period"]]*len(dataset)
        filename = ["X_l0.npy", "X_l1.npy", "X_l2.npy"]
        paths = [[f"{root}/{folder}/{d}/{p}/{f}" for f in filename] for d, p in zip(dataset, period)]
        for path, ds in zip(paths, dataset):
            X_l0_m = np.load(path[0]).mean(axis=0)
            X_l1_m = np.load(path[1]).mean(axis=0)
            X_l2_m = np.load(path[2]).mean(axis=0)

            X_l0_m_diff = normalize(abs_diff(X_l0_m[:, 0], X_l0_m[:, 1]))
            X_l1_m_diff = normalize(abs_diff(X_l1_m[:, 0], X_l1_m[:, 1]))
            X_l2_m_diff = normalize(abs_diff(X_l2_m[:, 0], X_l2_m[:, 1]))

            # plotting dependencies between sensor axes
            # TODO: korelovat s rychlostí a směrem větru???
            fig = plt.figure()
            plt.title(f"{ds}")
            plt.stem(X_l0_m[:, 0], X_l0_m[:, 1], use_line_collection=True, basefmt=" ", markerfmt="or", linefmt="none")
            plt.stem(X_l1_m[:, 0], X_l1_m[:, 1], use_line_collection=True, basefmt=" ", markerfmt="xg", linefmt="none")
            plt.stem(X_l2_m[:, 0], X_l2_m[:, 1], use_line_collection=True, basefmt=" ", markerfmt="+b", linefmt="none")
            plt.plot([0, 2.], [0, 2.])

            # plotting differences between values of sensor axes
            fig, ax = plt.subplots(3, 1)
            plt.suptitle(f"{ds}")
            ax[0].stem(X_l0_m_diff, use_line_collection=True, basefmt="-r", markerfmt=" ", linefmt="-r")
            ax[1].stem(X_l1_m_diff, use_line_collection=True, basefmt="-g", markerfmt=" ", linefmt="-g")
            ax[2].stem(X_l2_m_diff, use_line_collection=True, basefmt="-b", markerfmt=" ", linefmt="-b")

            plt.show()