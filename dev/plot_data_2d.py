import numpy as np
from matplotlib import pyplot as plt

from bin.helpers import console_logger

LOGGER = console_logger(__name__, level="DEBUG")

FS = 512  # Hz
NFFT = 5120

FREQS_PATH = "../data/generated/train/freqs.npy"
TRAIN_PATH = "../data/generated/train/raw0_sAmp(0, 1)_nAmp(0, 0.5).npy"
EVAL_PATH = "../data/generated/test/raw0_sAmp(0, 1)_nAmp(0, 0.5).npy"

TRAIN_COLOR = "blue"
UNDAMAGED_COLOR = "green"
DAMAGED_COLOR = "red"

if __name__ == '__main__':
    freqs = np.load(FREQS_PATH)
    train_data = np.load(TRAIN_PATH)
    eval_data = np.load(EVAL_PATH)

    LOGGER.debug(f"freqs.shape: {freqs.shape}")
    LOGGER.debug(f"train_data.shape: {train_data.shape}")
    LOGGER.debug(f"eval_data.shape: {eval_data.shape}")

    full_data = np.concatenate((train_data, eval_data), axis=0)
    full_data = (full_data - full_data.mean())/full_data.std()

    LOGGER.debug(f"full_data.shape: {full_data.shape}")

    fss = np.arange(NFFT // 2) / NFFT * FS
    X_full_data = abs(np.fft.fft(full_data, NFFT))[:, :NFFT // 2]

    x = np.arange(X_full_data.shape[0])
    y = freqs

    X, Y = np.meshgrid(x, y)
    Z = X_full_data.T

    LOGGER.debug(f"X.shape: {X.shape}")
    LOGGER.debug(f"Y.shape: {Y.shape}")
    LOGGER.debug(f"Z.shape: {Z.shape}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X[:, :11], Y[:, :11], Z[:, :11], rstride=50, cstride=256, color=TRAIN_COLOR)
    ax.plot_surface(X[:, 10:31], Y[:, 10:31], Z[:, 10:31], rstride=50, cstride=256, color=UNDAMAGED_COLOR)
    ax.plot_surface(X[:, 30:], Y[:, 30:], Z[:, 30:], rstride=50, cstride=256, color=DAMAGED_COLOR)
    ax.set_xlabel("sample")
    ax.set_ylabel("frequency (Hz)")
    ax.set_zlabel("abs(fft)")
    proxy_l1 = plt.plot([0, 0], [0, 0], color=TRAIN_COLOR)
    proxy_l2 = plt.plot([0, 0], [0, 0], color=UNDAMAGED_COLOR)
    proxy_l3 = plt.plot([0, 0], [0, 0], color=DAMAGED_COLOR)
    ax.legend(["train", "eval undamaged", "eval damaged"], loc="upper center")

#    ax = fig.add_subplot(212)
#    ax.pcolormesh(x, y, Z)
#    ax.set_xlabel("sample")
#    ax.set_ylabel("frequency (Hz)")

    plt.tight_layout(pad=2)
    plt.savefig("../data/generated/surface.png", dpi=300)
    plt.savefig("../data/generated/surface.pdf")
    plt.savefig("../data/generated/surface.svg")

    plt.show()
