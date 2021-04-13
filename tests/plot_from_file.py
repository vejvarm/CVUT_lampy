import os

from matplotlib import pyplot as plt
import numpy as np

from bin.flags import FLAGS

PATH_TO_FOLDER = os.path.join(FLAGS.data_root, FLAGS.preprocessed_folder, FLAGS.paths["serialized"]["folder"])
FILE_IDXS = range(100, 120)

if __name__ == '__main__':
    for idx in FILE_IDXS:
        files = [os.path.join(PATH_TO_FOLDER, f) for f in next(os.walk(PATH_TO_FOLDER))[-1]]

        try:
            freqs = np.load(os.path.join(PATH_TO_FOLDER, "freqs.npy"))
        except FileNotFoundError:
            freqs = None

        psd_day = np.load(files[idx])

        fig, axes = plt.subplots(3, 2)
        axes = [a for ax in axes for a in ax]
        print(len(axes))
        print(axes)
        for ax, psd in zip(axes, psd_day):
            ax.plot(freqs, psd.mean(-1))

        plt.show()
