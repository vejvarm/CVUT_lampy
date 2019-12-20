import numpy as np

from scipy.optimize import leastsq
from matplotlib import pyplot as plt

from preprocessing import Preprocessor


def curve_linear(par, x):
    return par[0]*x + par[1]


def curve_quadratic(par, x):
    return par[0]*x**2 + par[1]*x + par[2]


def error_func(par, x, y, func):
    # sum of squared of difference between func(par, x) and y
    return np.square(func(par, x) - y)


if __name__ == '__main__':
    filename = "08072018_AccM"
    path = ['../data/neporuseno/week/' + filename + '.mat']

    # preprocessing parameters (refer to Preprocessor __init__() for possible preprocessing settings):
    ns_per_hz = 10
    freq_range = (0, 256)
    noise_f_rem = (2, 50, 100, 150, 200)
    noise_df_rem = (2, 5, 2, 5, 2)
    mov_filt_size = 10

    preprocessor = Preprocessor(ns_per_hz=ns_per_hz,
                                freq_range=freq_range,
                                noise_f_rem=noise_f_rem,
                                noise_df_rem=noise_df_rem,
                                mov_filt_size=mov_filt_size)

    preprocessed = preprocessor.run(path)

    freqs, accs, wind_dirs, wind_spds = preprocessed[filename]

    # Curve fitting (Least Squares optimization)
    x = freqs
    y = accs[0][:, 0]

    # LINEAR
    params_lin_init = (1., 1.)

    print(x.shape, y.shape)

    params_linear, success = leastsq(error_func, params_lin_init, args=(x, y, curve_linear))

    print(params_linear)

    y_linear = curve_linear(params_linear, x)

    y_detrended_lin = y - y_linear

    # Quadratic
    params_quad_init = (1.0, 1.0, 1.0)
    params_quad, success = leastsq(error_func, params_quad_init, args=(x, y, curve_quadratic))

    y_quadratic = curve_quadratic(params_quad, x)

    y_detrended_quad = y - y_quadratic

    # PLOTTING
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(x, y)
    ax[0].plot(x, y_linear)
    ax[0].plot(x, y_quadratic)
    ax[1].plot(x, y_detrended_lin)
    ax[2].plot(x, y_detrended_quad)
    for a in ax:
        a.set_yscale('linear')
    plt.show()