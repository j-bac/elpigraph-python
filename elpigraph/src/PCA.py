# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 10:11:02 2018

@author: Alexis Martin
"""
import numpy as np

try:
    import cupy
except:
    pass
from scipy import linalg as la
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import check_random_state


def PCA(data):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    # mean center the data
    # data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = np.linalg.eigh(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors

    return evecs, data.dot(evecs), evals


def TruncPCA(X, n_components, algorithm="arpack"):
    svd = TruncatedSVD(algorithm=algorithm, n_components=n_components)
    prcomp = svd.fit_transform(X)
    s = svd.singular_values_
    Vt = svd.components_
    U = prcomp / s

    return prcomp, svd.explained_variance_, U, s, Vt


def PCA_gpu(data):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    # mean center the data
    # data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = cupy.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = cupy.linalg.eigh(R)
    # sort eigenvalue in decreasing order
    idx = cupy.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors

    return evecs, data.dot(evecs), evals


def TruncSVD_gpu(
    M, n_components, n_oversamples=10, n_iter="auto", transpose="auto", random_state=0
):
    """Computes a truncated randomized SVD on GPU. Adapted from Sklearn.
    Taken from : https://vip.readthedocs.io/en/latest/_modules/vip_hci/pca/svd.html

    Parameters
    ----------
    M : ndarray or sparse matrix
        Matrix to decompose
    n_components : int
        Number of singular values and vectors to extract.
    n_oversamples : int (default is 10)
        Additional number of random vectors to sample the range of M so as
        to ensure proper conditioning. The total number of random vectors
        used to find the range of M is n_components + n_oversamples. Smaller
        number can improve speed but can negatively impact the quality of
        approximation of singular vectors and singular values.
    n_iter : int or 'auto' (default is 'auto')
        Number of power iterations. It can be used to deal with very noisy
        problems. When 'auto', it is set to 4, unless `n_components` is small
        (< .1 * min(X.shape)) `n_iter` in which case is set to 7.
        This improves precision with few components.
    transpose : True, False or 'auto' (default)
        Whether the algorithm should be applied to M.T instead of M. The
        result should approximately be the same. The 'auto' mode will
        trigger the transposition if M.shape[1] > M.shape[0] since this
        implementation of randomized SVD tend to be a little faster in that
        case.
    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.
    lib : {'cp', 'pytorch'}, str optional
        Chooses the GPU library to be used.

    Notes
    -----
    This algorithm finds a (usually very good) approximate truncated
    singular value decomposition using randomization to speed up the
    computations. It is particularly fast on large matrices on which
    you wish to extract only a small number of components. In order to
    obtain further speed up, `n_iter` can be set <=2 (at the cost of
    loss of precision).

    References
    ----------
    * Finding structure with randomness: Stochastic algorithms for constructing
      approximate matrix decompositions
      Halko, et al., 2009 http://arxiv.org/abs/arXiv:0909.4061
    * A randomized algorithm for the decomposition of matrices
      Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert
    * An implementation of a randomized algorithm for principal component
      analysis
      A. Szlam et al. 2014
    """
    random_state = check_random_state(random_state)
    n_random = n_components + n_oversamples
    n_samples, n_features = M.shape

    if n_iter == "auto":
        # Checks if the number of iterations is explicitly specified
        n_iter = 7 if n_components < 0.1 * min(M.shape) else 4

    if transpose == "auto":
        transpose = n_samples < n_features
    if transpose:
        M = M.T  # this implementation is a bit faster with smaller shape[1]

    # Generating normal random vectors with shape: (M.shape[1], n_random)
    Q = random_state.normal(size=(M.shape[1], n_random))
    Q = cupy.array(Q)
    Q = cupy.asarray(Q)

    # Perform power iterations with Q to further 'imprint' the top
    # singular vectors of M in Q
    for i in range(n_iter):
        Q = cupy.dot(M, Q)
        Q = cupy.dot(M.T, Q)

    # Sample the range of M using by linear projection of Q. Extract an orthonormal basis
    Q, _ = cupy.linalg.qr(cupy.dot(M, Q), mode="reduced")

    # project M to the (k + p) dimensional space using the basis vectors
    B = cupy.dot(Q.T, M)

    B = cupy.array(B)
    Q = cupy.array(Q)
    # compute the SVD on the thin matrix: (k + p) wide
    Uhat, s, V = cupy.linalg.svd(B, full_matrices=False, compute_uv=True)
    del B
    U = cupy.dot(Q, Uhat)

    if transpose:
        # transpose back the results according to the input convention
        return V[:n_components, :].T, s[:n_components], U[:, :n_components].T
    else:
        return U[:, :n_components], s[:n_components], V[:n_components, :]
