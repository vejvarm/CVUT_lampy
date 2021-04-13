import os

from bin.Preprocessor import Preprocessor

from bin.flags import FLAGS

# dataset settings
SETTINGS = ["training", "validation", "serialized"]

# save to individual files into given folder:
NPY_SAVE_FOLDER = "g:/datasets/lamps/preprocessed/"

if __name__ == '__main__':

    settings = SETTINGS[-1:]
    print(settings)

    # path settings
    root = os.path.join(FLAGS.data_root, FLAGS.raw_folder)
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
            preprocessor.run([path], npy_save_folder=NPY_SAVE_FOLDER)

