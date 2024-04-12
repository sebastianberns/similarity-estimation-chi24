import numpy as np
from numpy.typing import NDArray

from .distance import pairwise_euclidean_distance
from .level import Level
from .metric import Metric
from .normalize import normalize


def pairwise_embedding_similarity(embedding: NDArray) -> NDArray:
    # Compute pairwise distances
    distances = pairwise_euclidean_distance(embedding)

    # Normalize distances to [0, 1]
    distances = normalize(distances)

    # Similarity is inverse of normalized distance
    similarities = 1 - distances
    return similarities


def pairwise_level_similarity(*levels: Level, rep: str, metric: Metric) -> NDArray:
    """ Calculate the pairwise similarity of a list of levels

        Arguments:
            levels (Level...):  Levels to compare
            metric (Metric):  Instance of a Metric class
        Returns (NDArray):  Pairwise similarity matrix (symmetric)
    """
    assert isinstance(metric, Metric), f"Invalid metric: {metric}"

    num_levels = len(levels)  # Number of levels
    similarities = np.zeros((num_levels, num_levels))  # Similarity matrix

    for i, level1 in enumerate(levels):
        for k, level2 in enumerate(levels[i:]):
            j = i + k  # Index of level2

            # Compute pairwise similarity
            similarity = metric.similarity(level1, level2, rep)

            # Fill in symmetric matrix
            similarities[i, j] = similarity
            similarities[j, i] = similarity

    # Normalize similarities to [0, 1]
    similarities = normalize(similarities)
    
    return similarities
