# if not period and bin_size == 5 and threshold == 0.1:
#     _, ax1 = plt.subplots()
#     ax2 = ax1.twinx()
#     ln1 = ax1.stem(freq_binarized_mean, PSD_binarized_softmax[0, :], markerfmt="r ", linefmt="r",
#             use_line_collection=True, label="binarizace spektra")
#     ln2 = ax2.plot(freq_binarized_mean, PSD_bins.mean(axis=-1)[0, :], "-b",
#             label="st�edn� hodnoty ko��k�", alpha=0.7)
#     ln2.append(ln1)
#     labs = [l.get_label() for l in ln2]
#     ax1.legend(ln2, labs, loc=0)
#     ax1.set_xlabel("frekvence (Hz)")
#     ax2.set_ylabel("psd_bins")
#     ax1.set_ylabel(f"psd_({bin_size}, {threshold})")
#     ax1.set_title(f"Porovn�n� spektra a jeho binarizace (bs={bin_size}, th={threshold})")
#     plt.savefig(f"./images/M2/{i}_bs-{bin_size}_th{threshold}_{np.random.randint(0,1000)}.pdf")