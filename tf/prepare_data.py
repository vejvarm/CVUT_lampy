from preprocessing import Preprocessor

import numpy as np


if __name__ == '__main__':
    folder = "validace"
    dataset = ["neporuseno", "poruseno"]
    period = ""  # week or 2months
    paths = [f"../data/{folder}/{d}/{period}" for d in dataset]

    nlamps = 3
    naccs_per_lamp = 2

    preprocessor = Preprocessor()

    for path in paths:
        freqs, psd_array = preprocessor.run([path], return_as="ndarray")

        ndays, naccs, nfft, nmeas = psd_array.shape
        # reshape from (ndays, naccs, nfft/2, nmeas) to (ndays*nmeas, nfft/2, naccs)
        psd_array = psd_array.transpose(0, 3, 2, 1).reshape(ndays*nmeas, nfft, naccs)

        # split into accs from different lamp posts (there are 3 lamps, each with 2 accs)
        psd_array_split = np.split(psd_array, nlamps, axis=-1)

        # stack split arrays on top of each other (axis 0)
        psd_array = np.vstack(psd_array_split)

        # generate labels (0 if neporuseno, 1 if poruseno)
        if "neporuseno" in path:
            labels = np.zeros((ndays*nmeas*nlamps, ), dtype=np.float32)
        else:
            # 1st lamp in poruseno is not broken (only 2nd and 3rd are broken) !!!
            labels = np.concatenate((np.zeros((ndays*nmeas, ), dtype=np.float32),
                                     np.ones((ndays*nmeas*(nlamps-1), ), dtype=np.float32)))

        print(psd_array.shape, labels.shape)

        # save data (X) and labels (y) to numpy arrays in respective paths
        np.save(path+"/X.npy", psd_array)
        np.save(path+"/y.npy", labels)

