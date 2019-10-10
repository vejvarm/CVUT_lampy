# Method 1 (Finding centres of mass in PSD and then comparing them)
import numpy as np

from os import path
from matplotlib import pyplot as plt

from preprocessing import Preprocessor
from helpers import find_top_peaks, calc_centre_of_mass


class M1:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def train(self, path):
        """ learn top peak indices and their centres of mass"""
        pass

    def compare(self):
        """ calculate centres of mass at the learned peak indices and compare them with the learned centres of mass"""
        pass


if __name__ == '__main__':
    dataset = ["neporuseno", "neporuseno2", "poruseno"]
    period = "2months"  # week or 2months
    paths = [f"./data/{d}/{period}" for d in dataset]
    freqs_path = "./data/freqs.npy"

    # preprocess files from neporuseno, neporuseno2 and poruseno
    preprocessor = Preprocessor()
    # load freqs and PSD if the preprocessed files already exist:
    if all(path.exists(p + "/PSD.npy") for p in paths):
        freqs = np.load("./data/freqs.npy")
        PSD_neporuseno = np.load(paths[0]+"/PSD.npy")
        PSD_neporuseno2 = np.load(paths[1]+"/PSD.npy")
        PSD_poruseno = np.load(paths[2]+"/PSD.npy")
    else:
        (freqs, psd_neporuseno), (_, psd_neporuseno2), (_, psd_poruseno) = [preprocessor.run([p], return_as="ndarray")
                                                                            for p in paths]
        # calculate PSD (== long term average values of psd)
        PSD_neporuseno = psd_neporuseno.mean(axis=(0, 3))
        PSD_neporuseno2 = psd_neporuseno2.mean(axis=(0, 3))
        PSD_poruseno = psd_poruseno.mean(axis=(0, 3))

        # save freqs and PSD files
        np.save(freqs_path, freqs)
        np.save(paths[0]+"/PSD.npy", PSD_neporuseno)
        np.save(paths[1]+"/PSD.npy", PSD_neporuseno2)
        np.save(paths[2]+"/PSD.npy", PSD_poruseno)

    # find peaks and learn centres of mass of neporuseno
    ns_per_hz = preprocessor.ns_per_hz
    delta_f = ns_per_hz*5
    peak_distance = delta_f*2
    n_peaks = 14

    peaks_neporuseno = find_top_peaks(PSD_neporuseno, peak_distance, n_peaks)
    centres_of_mass_neporuseno = calc_centre_of_mass(PSD_neporuseno, peaks_neporuseno, delta_f, ns_per_hz)

#    print(f"Learned Centres of Mass from neporuseno: {centres_of_mass_neporuseno}")

    # Comparing centres of mass of the learned peak frequencies with neporuseno2 and poruseno
    centres_of_mass_neporuseno2 = calc_centre_of_mass(PSD_neporuseno2, peaks_neporuseno, delta_f, ns_per_hz)
    centres_of_mass_poruseno = calc_centre_of_mass(PSD_poruseno, peaks_neporuseno, delta_f, ns_per_hz)

#    print(f"Centres of Mass from neporuseno2: {centres_of_mass_neporuseno2}")
#    print(f"Centres of Mass from poruseno: {centres_of_mass_poruseno}")

    # calculate differences
    diff_neporuseno2 = np.square(centres_of_mass_neporuseno - centres_of_mass_neporuseno2)
    diff_poruseno = np.square(centres_of_mass_neporuseno - centres_of_mass_poruseno)

#    print(f"Difference between neporuseno & neporuseno2: {diff_neporuseno2}")
#    print(f"Difference between neporuseno & poruseno: {diff_poruseno}")

    print(f"________##### {n_peaks} #####_________")
    print(f"Sum of square differences neporuseno2: {diff_neporuseno2.sum()}")
    print(f"Sum of square differences poruseno: {diff_poruseno.sum()}")
