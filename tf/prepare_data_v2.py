from preprocessing import Preprocessor

import numpy as np

from flags import FLAGS

if __name__ == '__main__':

    # physical settings
    nlamps = 3
    naccs_per_lamp = 6

    settings = ["training", "validation", "serialized"][-1:]
    print(settings)

    ndays_unbroken = FLAGS.serialized["unbroken"]
    ndays_broken = FLAGS.serialized["broken"]

    # path settings
    root = ".."
    # setting = "training"  # "training" or "validation"
    for setting in settings:
        folder = FLAGS.paths[setting]["folder"]
        dataset = FLAGS.paths[setting]["dataset"]
        period = FLAGS.paths[setting]["period"]  # week or 2months
        paths = [f"{root}/{folder}/{d}/{period}" for d in dataset]

        # preprocessing settings
        preprocessor = Preprocessor(use_autocorr=FLAGS.preprocessing["use_autocorr"],
                                    rem_neg=FLAGS.preprocessing["rem_neg"])

        for path in paths:
            freqs, psd_array = preprocessor.run([path], return_as="ndarray")

            ndays, naccs, nfft, nmeas = psd_array.shape
            # reshape from (ndays, naccs, nfft/2, nmeas) to (ndays*nmeas, nfft/2, naccs)
            psd_array = psd_array.transpose(0, 3, 2, 1).reshape(ndays*nmeas, nfft, naccs)

            # split into accs from different lamp posts (there are 3 lamps, each with 2 accs)
            psd_array_split = np.split(psd_array, nlamps, axis=-1)

            for i, psd_array in enumerate(psd_array_split):
                # generate labels (0 if neporuseno, 1 if poruseno)
                if "neporuseno" in path:
                    labels = np.zeros((ndays*nmeas, ), dtype=np.float32)
                elif i == 0:
                    # 1st lamp in is never broken (only 2nd and 3rd are broken) !!!
                    labels = np.zeros((ndays*nmeas, ), dtype=np.float32)
                elif "poruseno" in path:
                    # 2nd and 3rd lamp from poruseno are broken
                    labels = np.ones((ndays*nmeas, ), dtype=np.float32)
                elif "serialized" in setting:
                    # for serialized scenario
                    print("running scenario for serialized folder")
                    ndays_unbroken = 86
                    ndays_broken = 140
                    labels = np.concatenate((np.zeros((ndays_unbroken*nmeas, ), dtype=np.float32),
                                             np.ones((ndays_broken*nmeas, ), dtype=np.float32)))
                else:
                    raise NameError("Folder structure doesn't fit the schema.")

                print(f"X_l{i}: {psd_array.shape}\n y_l{i}: {labels.shape}")

                # save data (X) and labels (y) to numpy arrays in respective paths
                np.save(path+f"/X_l{i}.npy", psd_array)
                np.save(path+f"/y_l{i}.npy", labels)

