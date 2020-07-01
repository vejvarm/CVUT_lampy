from matplotlib import pyplot as plt

from bin.Preprocessor import Preprocessor
# from helpers import find_top_peaks, calc_centre_of_mass


if __name__ == '__main__':
    ns_per_hz = 10
    delta_f = ns_per_hz*5
    n_peaks = 10

    path = ["../data/poruseno/week"]

    preprocessor = Preprocessor()
    freqs, psd_stacked = preprocessor.run(path, return_as='ndarray')

    print(psd_stacked.shape)

    psd_average = psd_stacked.mean(axis=(0, 3))

    # Finding (learning) peaks
    top_sorted_peaks = find_top_peaks(psd_average, 2*delta_f, n_peaks)

    print(psd_average.shape)
    print(top_sorted_peaks.shape)

    # Calculating centre of mass (těžiště)
    peak_com_arr = calc_centre_of_mass(psd_average, top_sorted_peaks, delta_f)

    print(f"Top sorted peaks: {top_sorted_peaks}")
    print(f"Peak centre of mass: {peak_com_arr}")


    # Plotting
    fig, axes = plt.subplots(3, 2)
    for i, ax in enumerate(axes.flatten()):
        ax.plot(freqs, psd_average[i, :])
        ax.stem(freqs[top_sorted_peaks[i, :]], psd_average[i, top_sorted_peaks[i, :]], '-rx', use_line_collection=True)
        ax.stem(freqs[peak_com_arr[i, :]], psd_average[i, peak_com_arr[i, :]], '-gx', use_line_collection=True)
    plt.show()
