import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import cohen_kappa_score, confusion_matrix  # type: ignore[import]


def cohen_kappa_max(y1: NDArray, y2: NDArray, categories=None, weights=None, sample_weight=None) -> float:
    # https://github.com/scikit-learn/scikit-learn/blob/7f9bad99d/sklearn/metrics/_classification.py#L614
    confusion = confusion_matrix(y1, y2, labels=categories, sample_weight=sample_weight)
    n_classes = confusion.shape[0]
    assert n_classes == 2, "Currently only supports two categories"

    sum0 = np.sum(confusion, axis=0)  # Column marginals
    sum1 = np.sum(confusion, axis=1)  # Row marginals

    # Find minimum value for either category
    min0 = min(sum0[0], sum1[0])
    min1 = min(sum0[1], sum1[1])

    # Adjust confusion matrix according to maximum achievable agreement
    confusion = np.array([
        [min0, sum1[0] - min0],
        [sum0[0] - min0, min1]
    ])

    # Recalculate with updated confusion matrix
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)

    if weights is None:
        w_mat = np.ones([n_classes, n_classes], dtype=int)
        w_mat.flat[:: n_classes + 1] = 0
    else:  # "linear" or "quadratic"
        w_mat = np.zeros([n_classes, n_classes], dtype=int)
        w_mat += np.arange(n_classes)
        if weights == "linear":
            w_mat = np.abs(w_mat - w_mat.T)
        else:
            w_mat = (w_mat - w_mat.T) ** 2

    k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
    return 1 - k


def quantity_disagreement(y1: NDArray, y2: NDArray, categories=None) -> float:
    confusion = confusion_matrix(y1, y2, labels=categories)
    n_catgories = confusion.shape[0]
    assert n_catgories == 2, "Currently only supports two categories"

    total = np.sum(confusion)  # Total number of agreements
    probabilities = confusion / total

    sumc = np.sum(probabilities, axis=0)  # Column marginals
    sumr = np.sum(probabilities, axis=1)  # Row marginals
    sums = np.vstack((sumc, sumr))

    q = np.abs(np.diff(sums, axis=0))  # Quantity agreement per category
    return np.sum(q) / 2  # Overall quantity agreement


def allocation_disagreement(y1: NDArray, y2: NDArray, categories=None) -> float:
    confusion = confusion_matrix(y1, y2, labels=categories)
    n_catgories = confusion.shape[0]
    assert n_catgories == 2, "Currently only supports two categories"

    total = np.sum(confusion)  # Total number of agreements
    probabilities = confusion / total

    diag = np.diag(probabilities)  # Agreement per category
    sumc = np.sum(probabilities, axis=0)  # Column marginals
    sumr = np.sum(probabilities, axis=1)  # Row marginals
    sums = np.vstack((sumc, sumr))

    a = 2 * np.min(sums - diag, axis=0)  # Allocation agreement per category
    return np.sum(a) / 2  # Overall allocation agreement
