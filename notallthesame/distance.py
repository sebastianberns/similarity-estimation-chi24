"""
Distance functions between
    - scalars
    - embedding vectors
    - distributions
"""
from typing import Optional, Union
import warnings

import numpy as np
from numpy.typing import NDArray


def pairwise_euclidean_distance(X: NDArray, Y: Optional[NDArray] = None, verbose: bool = True) -> NDArray:
    """ Compute pairwise distances between two collections of items or between the items of one collection

        X (NDArray): feature matrix [N x D]
        Y (NDArray): optional second feature matrix [N x D]

        Return (NDArray): Symmetric pairwise distance matrix [N x N]
    """
    if Y is None:
        Y = X

    # Check basic assumptions
    assert len(X.shape) == 2  # Is matrix
    assert len(Y.shape) == 2
    assert X.shape[0] == Y.shape[0]  # Same size
    assert X.shape[1] == Y.shape[1]

    # Helper function
    def squared_norm(A):
        num = A.shape[0]
        A = A.astype(np.float64)  # to prevent underflow
        A = np.sum(A**2, axis=1, keepdims=True)
        A = np.repeat(A, num, axis=1)
        return A

    # Squared norms
    X2 = squared_norm(X)
    Y2 = squared_norm(Y).T

    XY = np.dot(X, Y.T)  # Gram matrix
    D2 = X2 - 2*XY + Y2  # Euclidean distance matrix

    # check negative distance
    negative = D2 < 0  # Indices of entries below zero
    if negative.any():  # Are there any negative squared distances?
        D2[negative] = 0.  # Set to zero
        if verbose:
            warnings.warn(f"{negative.sum()} negative squared distances found and set to zero")

    distances = np.sqrt(D2)  # Actual distances
    return distances


def relative_entropy(p: NDArray, q: NDArray) -> Union[float, NDArray]:
    assert p.shape == q.shape, "Distributions must have the same shape"

    entropy = np.zeros_like(p)  # Initialize entropy, leave zero where p is zero
    entropy[q == 0] = np.inf  # Set entropy to infinity where q is zero

    mp = p != 0  # Mask of non-zero values in p
    mq = q != 0  # Mask of non-zero values in q
    m = mp * mq  # Joint mask of non-zero values in p and q

    entropy[m] = p[m] * np.log2(p[m] / q[m])
    return entropy


def kl_divergence(p: NDArray, q: NDArray) -> float:
    """ Calculate the Kullback-Leibler divergence between two distributions
        Using log base 2, result is in bits

        p, q (NDArray): Distributions
        Returns (float): Kullback-Leibler divergence between p and q

        No upper bound on KL divergence with log base 2
    """
    return np.sum(relative_entropy(p, q))


def js_divergence(p: NDArray, q: NDArray) -> float:
    """ Calculate the Jensen-Shannon divergence between two distributions

        p, q (NDArray): Distributions
        Returns (float): Jensen-Shannon divergence between p and q

        Upper bound on JS divergence is 1
    """
    m = (p + q) / 2.0
    js = (kl_divergence(p, m) + kl_divergence(q, m)) / 2.0
    return js


def js_distance(p: NDArray, q: NDArray) -> float:
    """ Calculate the Jensen-Shannon distance between two distributions

        p, q (NDArray): Distributions
        Returns (float): Jensen-Shannon distance between p and q

        Upper bound on JS distance is 1
    """
    js = js_divergence(p, q)
    return np.sqrt(js)
