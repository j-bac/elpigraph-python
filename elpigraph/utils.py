import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import networkx as nx
from sklearn.neighbors import NearestNeighbors, KernelDensity


def _longform_knn_to_sparse(dis, idx):
    row_ind = np.tile(np.arange(len(idx))[:, None], idx.shape[1])
    col_ind = idx
    return scipy.sparse.csr_matrix((dis.flat, (row_ind.flat, col_ind.flat)))


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


def ordinal_neighbors_stagewise_longform(
    X, stages_labels, stages=None, k=15, radius=None, m="cosine"
):
    """Supervised (ordinal) nearest-neighbor search.
    Stages is an ordered list of stages labels (low to high). If None, taken as np.unique(stages_labels)"""

    if stages is None:
        stages = np.unique(stages_labels)

    knn_distances = np.zeros((len(X), 3 * k))
    knn_idx = np.zeros((len(X), 3 * k), dtype=int)

    nn_stage = {}
    for s in stages:
        nn_stage[s] = NearestNeighbors(n_neighbors=k, metric=m).fit(
            X[stages_labels.values == s, :]
        )

    s = []
    t = []
    w = []
    for i in range(len(stages) - 1):
        dis, ind = nn_stage[stages[i]].kneighbors(
            X[stages_labels.values == stages[i + 1], :], k
        )
        knn_distances[stages_labels.values == stages[i + 1], :k] = dis
        knn_idx[stages_labels.values == stages[i + 1], :k] = np.where(
            stages_labels.values == stages[i]
        )[0][ind]

    for i in range(1, len(stages)):
        dis, ind = nn_stage[stages[i]].kneighbors(
            X[stages_labels.values == stages[i - 1], :], k
        )
        knn_distances[stages_labels.values == stages[i - 1], k : 2 * k] = dis
        knn_idx[stages_labels.values == stages[i - 1], k : 2 * k] = np.where(
            stages_labels.values == stages[i]
        )[0][ind]

    for i in range(len(stages)):
        if i == 0:
            dis, ind = nn_stage[stages[i]].kneighbors(
                X[stages_labels.values == stages[i], :], 2 * k + 1
            )
            dis, ind = dis[:, 1:], ind[:, 1:]
            knn_distances[
                np.argwhere(stages_labels.values == stages[i]),
                list(range(k)) + list(range(2 * k, 3 * k)),
            ] = dis
            knn_idx[
                np.argwhere(stages_labels.values == stages[i]),
                list(range(k)) + list(range(2 * k, 3 * k)),
            ] = np.where(stages_labels.values == stages[i])[0][ind]

        elif i == (len(stages) - 1):
            dis, ind = nn_stage[stages[i]].kneighbors(
                X[stages_labels.values == stages[i], :], 2 * k + 1
            )
            dis, ind = dis[:, 1:], ind[:, 1:]
            knn_distances[stages_labels.values == stages[i], k:] = dis
            knn_idx[stages_labels.values == stages[i], k:] = np.where(
                stages_labels.values == stages[i]
            )[0][ind]

        else:
            dis, ind = nn_stage[stages[i]].kneighbors(
                X[stages_labels.values == stages[i], :], k + 1
            )
            dis, ind = dis[:, 1:], ind[:, 1:]
            knn_distances[stages_labels.values == stages[i], 2 * k :] = dis
            knn_idx[stages_labels.values == stages[i], 2 * k :] = np.where(
                stages_labels.values == stages[i]
            )[0][ind]

    _sort = np.argsort(knn_distances, axis=1)
    knn_distances = knn_distances[np.arange(len(knn_distances))[:, None], _sort]
    knn_idx = knn_idx[np.arange(len(knn_distances))[:, None], _sort]

    return knn_distances, knn_idx


def ordinal_neighbors_longform(X, stages_labels, stages=None, k=15, m="cosine"):
    """Supervised (ordinal) nearest-neighbor search.
    Stages is an ordered list of stages labels (low to high). If None, taken as np.unique(stages_labels)"""
    if stages is None:
        stages = np.unique(stages_labels)

    knn_distances = np.zeros((len(X), k))
    knn_idx = np.zeros((len(X), k), dtype=int)
    for i in range(len(stages)):

        sel_points = (
            (stages_labels.values == stages[i])
            | (stages_labels.values == stages[max(0, i - 1)])
            | (stages_labels.values == stages[min(i + 1, len(stages) - 1)])
        )

        stage = stages_labels.values == stages[i]
        dis, ind = (
            NearestNeighbors(n_neighbors=k, metric=m)
            .fit(X[sel_points, :])
            .kneighbors(X[stage, :])
        )

        knn_distances[stage, :] = dis
        knn_idx[stage, :] = np.where(sel_points)[0][ind]
    return knn_distances, knn_idx


def supervised_knn(
    X,
    stages_labels,
    stages=None,
    n_neighbors=15,
    n_natural=0,
    m="cosine",
    method="force",
    return_sparse=False,
):
    """Supervised (ordinal) nearest-neighbor search.
    Stages is an ordered list of stages labels (low to high). If None, taken as np.unique(stages_labels)

    Parameters
    ----------
    method : str (default='force')
        if 'force', searches for each point at stage[i] n_neighbors nearest_neighbors, forcing:
            - n_neighbors/3 to be from stage[i-1]
            - n_neighbors/3 to be from stage[i]
            - n_neighbors/3 to be from stage[i+1]
            For stage[0] and stage[-1], 2*n_neighbors/3 are taken from stage[i]

        if 'guide', searches for each point at stage[i] n_neighbors nearest_neighbors
            from points in {stage[i-1], stage[i], stage[i+1]}, without constraints on proportions
    """

    if n_neighbors % 3 != 0:
        raise ValueError("Please provide n_neighbors divisible by 3")
    if stages is None:
        stages = np.unique(stages_labels)

    dis, idx = (
        NearestNeighbors(n_neighbors=n_neighbors, metric=m, n_jobs=8)
        .fit(X)
        .kneighbors()
    )

    if method == "guide":
        knn_distances, knn_idx = ordinal_neighbors_longform(
            X, stages_labels, stages=stages, k=n_neighbors, m=m
        )
    if method == "force":
        knn_distances, knn_idx = ordinal_neighbors_stagewise_longform(
            X, stages_labels, stages=stages, k=n_neighbors // 3, m=m
        )

    # ---mix natural nn with ordinal nn
    merged_idx = np.zeros((len(X), n_neighbors), dtype=np.int32)
    merged_dists = np.zeros((len(X), n_neighbors))
    for i in range(len(X)):
        merged_idx[i][:n_natural] = idx[i][:n_natural]
        merged_idx[i][n_natural:] = np.setdiff1d(
            knn_idx[i], idx[i][:n_natural], assume_unique=True
        )[: n_neighbors - n_natural]

        merged_dists[i][:n_natural] = dis[i][:n_natural]
        merged_dists[i][n_natural:] = knn_distances[i][
            np.isin(
                knn_idx[i],
                np.setdiff1d(knn_idx[i], idx[i][:n_natural], assume_unique=True)[
                    : n_neighbors - n_natural
                ],
            )
        ]

    if return_sparse:
        return _longform_knn_to_sparse(merged_dists, merged_idx)
    else:
        return merged_dists, merged_idx


def geodesic_pseudotime(X, k, root, g=None):
    """pseudotime as graph distance from root point"""
    if g is None:
        nn = NearestNeighbors(n_neighbors=k, n_jobs=8).fit(X)
        g = nx.convert_matrix.from_scipy_sparse_matrix(
            nn.kneighbors_graph(mode="distance")
        )
    else:
        g = nx.convert_matrix.from_scipy_sparse_matrix(g)
    if len(list(nx.connected_components(g))) > 1:
        raise ValueError(
            f"detected more than 1 components with k={k} neighbors. Please increase k"
        )
    lengths = nx.single_source_dijkstra_path_length(g, root)
    pseudotime = np.array(pd.Series(lengths).sort_index())
    return pseudotime