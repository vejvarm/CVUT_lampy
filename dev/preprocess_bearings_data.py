# data from: https://csegroups.case.edu/bearingdatacenter/pages/download-data-file
import os

import numpy as np
import scipy.io

from matplotlib import pyplot as plt

from dev.helpers import console_logger
from preprocessing import Preprocessor
from Methods import M2

LOGGER = console_logger(__name__, "DEBUG")

def prepare_data(paths, bearing):

    data_dict = {}
    for path in paths:
        data = scipy.io.loadmat(path)
        for key in data.keys():
            if f"{bearing}_time" in key:
                name = os.path.splitext(path.split("/")[-1])[0]
                num = int(''.join(c for c in key if c.isdigit()))
                data_dict[name] = [num, data]
                break
        else:
            print(f"no '{bearing}_time' in keys")
    names = list(data_dict.keys())

    x_list = []
    for name in names:
        num, data = data_dict[name]
        x = (data[f"X{num:03d}_{bearing}_time"])
        x = zscore(x.flatten())
        x_list.append((name, x))
    return data_dict, x_list


def zscore(x):
    return (x - x.mean())/(x.std())


def preprocess(x_list, p=Preprocessor()):
    freqs = None
    psd_list = []
    for name, x in x_list:
        arr = np.expand_dims(x, axis=1)
        freqs, psd = p.simple_preprocess(arr)
        psd_list.append(psd)
    return freqs, np.hstack(psd_list)


def expand_and_transpose(x):
    x = np.expand_dims(x, axis=0)
    return x.transpose((2, 1, 0))


if __name__ == '__main__':
    fs = 12000
    ns_per_hz = 1
    freq_range = (0, 6000)
    tdf_order = 3
    tdf_ranges = ((45, 55), )
    use_autocorr = False  # not working for this data (shape problems)
    rem_neg = True

    ntrain = 4

    bearing = "DE"  # "FE" or "DE"
    fault = "IR"  # "IR", "BL", "ORat06", "ORat03", "ORat12"
    if "DE" in bearing:
        root = "../data/drive_end/"
    elif "FE" in bearing:
        root = "../data/fan_end/"
    else:
        raise(ValueError, "Bearing choice must be either Drive End ('DE') or Fan End ('FE')")

    if "IR" in fault:
        root = root+"IR/"
    elif "ORat06" in fault:
        root = root+"ORat06/"
    elif "ORat03" in fault:
        root = root+"ORat03/"
    elif "ORat12" in fault:
        root = root+"ORat12/"
    elif "BL" in fault:
        root = root+"BL/"
    else:
        raise(ValueError, "Fault choice must be one of Inner Rail ('IR'), Outer Rail ('ORat##'), or  Ball ('BL')")

    paths = [f"{folder}{file}" for folder, _, file_list in os.walk(root) for file in file_list if ".mat" in file]

    train_paths = paths[:ntrain]
    test_paths = paths[ntrain:]

    train_save_folder = root+"train/"
    test_save_folder = root+"test/"

    LOGGER.info("LOADING FILES FROM PATHS")
    data_dict_train, x_list_train = prepare_data(train_paths, bearing)
    data_dict_test, x_list_test = prepare_data(test_paths, bearing)

    LOGGER.info("PREPROCESSING")
    p = Preprocessor(fs=fs, ns_per_hz=ns_per_hz, freq_range=freq_range, tdf_order=tdf_order, tdf_ranges=tdf_ranges,
                     use_autocorr=use_autocorr, rem_neg=rem_neg)
    freqs_train, psd_train = preprocess(x_list_train, p)
    freqs_test, psd_test = preprocess(x_list_test, p)

    LOGGER.debug(f"freqs equal: {np.all(freqs_train == freqs_test)}")
    LOGGER.debug(f"psd_train.shape: {psd_train.shape}")
    LOGGER.debug(f"psd_test.shape: {psd_test.shape}")

    LOGGER.info("RESHAPE psd_train and psd_test to (ndays, nfreqs, nmeas)")
    psd_train = expand_and_transpose(psd_train)
    psd_test = expand_and_transpose(psd_test)
    LOGGER.debug(f"psd_train.shape: {psd_train.shape}")
    LOGGER.debug(f"psd_test.shape: {psd_test.shape}")

    LOGGER.info("SAVE to .npy files")
    os.makedirs(train_save_folder, exist_ok=True)
    os.makedirs(test_save_folder, exist_ok=True)
    np.save(train_save_folder+"freqs.npy", freqs_train)
    np.save(train_save_folder+"X.npy", psd_train)
    np.save(test_save_folder+"freqs.npy", freqs_test)
    np.save(test_save_folder+"X.npy", psd_test)


    # TODO: Train M2 on psd_train
    # TODO: Evaluate on psd__test