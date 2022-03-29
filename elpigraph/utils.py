import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import networkx as nx
from sklearn.neighbors import NearestNeighbors, KernelDensity


def getWeights(X, bandwidth=1, griddelta=100, exponent=1, method="sklearn", **kwargs):
    """Get point weights as the inverse density of data
    X: np.array, (n_sample x n_dims)
    bandwidth: sklearn KernelDensity bandwidth if method == 'sklearn'
    griddelta: FFTKDE grid step size if method =='fft'
    exponent: density values are raised to the power of exponent
    """
    if method == "sklearn":
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth, **kwargs).fit(X)
        scores = kde.score_samples(X)
        scores = np.exp(scores)[:, None]

    elif method == "fft":
        import KDEpy

        kde = KDEpy.FFTKDE(**kwargs).fit(X)
        x, y = kde.evaluate(griddelta)
        scores = scipy.interpolate.griddata(x, y, X)

    p = 1 / (scores ** exponent)
    p /= np.sum(p)
    return p