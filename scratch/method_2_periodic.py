from flags import FLAGS
from preprocessing import Preprocessor
from Methods import M2

# TODO: bin statistics

if __name__ == "__main__":
    setting = "training"
    folder = FLAGS.paths[setting]["folder"]
    dataset = FLAGS.paths[setting]["dataset"]
    period = [FLAGS.paths[setting]["period"]]*len(dataset)
    filename = ["X.npy"]*len(dataset)
    paths = [f"../{folder}/{d}/{p}/{f}" for d, p, f in zip(dataset, period, filename)]

    print(paths)

    preprocessor = Preprocessor()

    from_existing_file = True
    var_scaled_PSD = False

    m2 = M2(preprocessor, var_scaled_PSD=var_scaled_PSD)

    # multiscale params
    bin_sizes = (10, 20, 40, 80)
    thresholds = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    plot_distributions = False

    # Train the method on 2 months of neporuseno
    m2.train(paths[0], bin_sizes, thresholds)

    ce2 = m2.compare(paths[1], period=1, print_results=False)
    ce3 = m2.compare(paths[2], period=1, print_results=False)

    #    print(ce2.shape)  # (nperiods, nbins*nthresholds)

    ce_diff = ce3 - ce2

    nbins = len(bin_sizes)
    nth = len(thresholds)

    for i, day in enumerate(ce_diff):
        daily_best = 0
        for j, val in enumerate(day):
            if val > daily_best:
                daily_best = val
                best_j = j
        print(f"day: {i}, bs: {bin_sizes[best_j % nbins]}, th: {thresholds[best_j % nth]} val: {daily_best}")