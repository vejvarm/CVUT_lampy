# short/long time averaging
# plotting
import os

from matplotlib import pyplot as plt

from bin.Preprocessor import Preprocessor
from bin.flags import FLAGS
from bin.Methods import M2

if __name__ == '__main__':
    setting = "training"
    folder = FLAGS.paths[setting]["folder"]
    dataset = FLAGS.paths[setting]["dataset"]
    period = ["2months"]*len(dataset)
    filename = ["X.npy", "", ""]
    paths = [f"./{folder}/{d}/{p}/{f}" for d, p, f in zip(dataset, period, filename)]

    # paths[0] == train path
    # paths[1] == validation undamaged
    # paths[2] == validation damaged

    undamaged_gen = os.walk(paths[1])
    damaged_gen = os.walk(paths[2])

    paths_undamaged = [os.path.join(folder, file) for folder, _, files in undamaged_gen for file in files]
    paths_damaged = [os.path.join(folder, file) for folder, _, files in damaged_gen for file in files]

    print(paths_undamaged)
    print(paths_damaged)

    preprocessor = Preprocessor(use_autocorr=FLAGS.preprocessing["use_autocorr"],
                                rem_neg=FLAGS.preprocessing["rem_neg"])

    from_existing_file = True
    var_scaled_PSD = False

    # METHOD 2 ---------------------------------------------------------------------------------------------------------
    m2 = M2(preprocessor=preprocessor, var_scaled_PSD=var_scaled_PSD)

    # multiscale params
    bin_sizes = (40, )
    tresholds = (0.5, )
    plot_distributions = True

    # Get distribution of training data
    distributions_1 = m2.get_multiscale_distributions(paths[0], bin_sizes=bin_sizes, thresholds=tresholds)

    ce2 = list()
    ce3 = list()
    ce_diff = list()
    print("\n--CROSS ENTROPY VALUES--")
    print(f"| day | bs | trsh || ds2 | ds3 | diff |")
    for i in range(len(paths_undamaged)):
        distributions_2 = m2.get_multiscale_distributions(paths_undamaged[i], bin_sizes=bin_sizes, thresholds=tresholds)
        distributions_3 = m2.get_multiscale_distributions(paths_damaged[i], bin_sizes=bin_sizes, thresholds=tresholds)
        for ((bin_sz, th), _, d1), ((_, _), _, d2), ((_, _), _, d3) in zip(distributions_1, distributions_2, distributions_3):
            ce2.append(M2._cross_entropy(d1, d2))
            ce3.append(M2._cross_entropy(d1, d3))
            ce_diff.append(ce3[i] - ce2[i])
            print(f'| {i:3d} | {bin_sz:2d} | {th:.2f} || {ce2[i]:3.0f} | {ce3[i]:3.0f} | {ce_diff[i]:4.0f} |')

    # plot crossentropy evolution in time
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ce2, "-b")
    ax[0].plot(ce3, "-.r")
    ax[0].set_title("Cross entropy of damaged and undamaged data")
    ax[0].set_ylabel("cross-entropy")
    ax[0].legend(("undamaged", "damaged"))

    ax[1].stem(ce_diff)
    ax[1].set_title("Difference in cross entropy between damaged and undamaged data")
    ax[1].set_ylabel("ce3 - ce2")
    for axis in ax:
        axis.set_xlabel("day")

    plt.show()

    # plotting distributions
#    if plot_distributions:
#        for j, dist in enumerate((distributions_1, distributions_2, distributions_3)):
#            for (params, freqs, d) in dist:
#                fig, axes = plt.subplots(3, 2)
#                for i, ax in enumerate(axes.flatten()):
#                    ax.semilogy(freqs, d[i, :])
#                fig.suptitle(f"{dataset[j]} | {params} |")
#        plt.show()

    # ------------------------------------------------------------------------------------------------------------------
