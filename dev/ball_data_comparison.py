# data from: https://csegroups.case.edu/bearingdatacenter/pages/download-data-file
import os

import numpy as np
import scipy.io

from matplotlib import pyplot as plt


def zscore(x):
    return (x - x.mean())/(x.std())


if __name__ == '__main__':
    bearing = "FE"  # "FE" or "DE"
    fault = "BL"  # "IR", "OR" or "BL"
    if "DE" in bearing:
        root = "../data/drive_end/"
    elif "FE" in bearing:
        root = "../data/fan_end/"
    else:
        raise(ValueError, "Bearing choice must be either Drive End ('DE') or Fan End ('FE')")

    if "IR" in fault:
        root = root+"IR/"
    if "OR" in fault:
        root = root+"OR/"
    if "BL" in fault:
        root = root+"BL/"
    else:
        raise(ValueError, "Fault choice must be one of Inner Rail ('IR'), Outer Rail ('OR'), or  Ball ('BL')")

    paths = [f"{folder}{file}" for folder, _, file_list in os.walk(root) for file in file_list]

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

    # DE ... Drive End bearing
    # FE ... Fan End bearing
    # BA ... Base Accelerometer
    x_list = []
    for name in names:
        num, data = data_dict[name]
        x = (data[f"X{num:03d}_{bearing}_time"])
    #                              data[f"X{num:03d}_FE_time"],
    #                              data[f"X{num:03d}_BA_time"],
    #                              data[f"X{num:03d}RPM"])

        x = zscore(x.flatten())
        x_list.append((name, x))

    # FFT
    fs = 12000  # sampling_frequency (Hz)
    nfft = 5120
    freqs = np.linspace(0, fs, nfft//2)

    X007 = []
    X014 = []
    X021 = []
    X028 = []

    for name, x in x_list:
        X = abs(np.fft.fft(x, nfft))[:nfft//2]
        if "007" in name:
            X007.append(X)
        elif "014" in name:
            X014.append(X)
        elif "021" in name:
            X021.append(X)
        elif "028" in name:
            X028.append(X)
        else:
            print(f"{name} not categorical")

    X007 = np.array(X007)
    X014 = np.array(X014)
    X021 = np.array(X021)
    X028 = np.array(X028)

    fig, ax = plt.subplots(4, 1)
    plt.suptitle("Drive End Bearing" if "DE" in bearing else "Fan End Bearing")
    ax[0].plot(freqs, X007.mean(axis=0), label=name)
    ax[1].plot(freqs, X014.mean(axis=0), label=name)
    ax[2].plot(freqs, X021.mean(axis=0), label=name)
    try:
        ax[3].plot(freqs, X028.mean(axis=0), label=name)
    except ValueError:
        print("No 0.28 inch data available")
    ax[0].set_title("0.07 inch")
    ax[1].set_title("0.14 inch")
    ax[2].set_title("0.21 inch")
    ax[3].set_title("0.28 inch")
    ax[3].set_xlabel("frequency [Hz]")
    fig.tight_layout()


