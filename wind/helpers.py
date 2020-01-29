import numpy as np

from matplotlib import pyplot as plt

def mapper(s, theta, d):
    for k, v in d.items():
        if k in s:
            theta.append(v)
            s = s.replace(k, "")
    return s, theta


def wd_str2rad(s):
    """ Map WindDirection string into a radian value for polar graph

    :param s: (string) Wind Direction string compounded from "N, S, E, W" letters
    """

    pi = np.pi
    trigrams = {"ENE": pi/8, "ESE": 15*pi/8}
    bigrams = {"NE": pi/4, "NW": 3*pi/4, "SW": 5*pi/4, "SE": 7*pi/4}
    unigrams = {"E": 0., "N": pi/2, "W": pi, "S": 3*pi/2}

    # map chars to numbers
    theta = []
    s, theta = mapper(s, theta, trigrams)
    s, theta = mapper(s, theta, bigrams)
    s, theta = mapper(s, theta, unigrams)

    return np.mean(theta)


if __name__ == '__main__':
    # TEST for wind dir
    ss = ["E", "ENE", "NE", "NNE", "N", "NNW", "NW", "WNW", "W", "WSW", "SW", "SSW", "S", "SSE", "SE", "ESE", "E"]
    x = np.linspace(1, 10, len(ss))

    thetas = np.array([wd_str2rad(s) for s in ss])

    thetas_deg = 180*thetas/np.pi
    print(thetas_deg)

    plt.polar(thetas, x, "rx")
    plt.show()
