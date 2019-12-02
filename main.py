# short/long time averaging
# TODO: plotting

import numpy as np

from matplotlib import pyplot as plt

from preprocessing import Preprocessor
from Aggregators import Average, complete_average


if __name__ == '__main__':
    root = "."
    folder = "trening"
    dataset = ["neporuseno", "neporuseno2",  "poruseno"]
    period = "2months"  # week or 2months
    file = "X.npy"
    paths = [f"{root}/data/{folder}/{d}/{period}/{file}" for d in dataset]

    print(paths)

    p = Preprocessor()

    freqs, psd_nepor = p.run([paths[0]], return_as="ndarray")

    print(psd_nepor.shape)


