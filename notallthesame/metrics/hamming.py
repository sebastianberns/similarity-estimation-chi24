import numpy as np
from numpy.typing import NDArray

from ..level import Level
from ..metric import Metric


class HammingDistance(Metric):
    def __init__(self) -> None:
        super().__init__("Hamming Distance")

    def compare(self, a: NDArray, b: NDArray) -> float:
        """ Calculate the Hamming distance of two levels
            Tiles types are compared directly
            Distance is the ratio of same tiles to total tiles

            Arguments:
                a, b (NDArray):  Levels to compare
            Returns (float): ratio of identical tiles to total tiles
        """
        assert a.shape == b.shape, "Levels must be the same shape"

        mismatch = a != b  # Compare level tiles
        ratio = np.sum(mismatch) / a.size  # Ratio of mismatches to total number of values

        return ratio

    def similarity(self, a: Level, b: Level, rep: str) -> float:
        """ Calculate the Hamming distance of two levels
            Tiles types are compared directly
            Distance is the ratio of same tiles to total tiles

            Arguments:
                a, b (Level):  Levels to compare
                rep (str):  Type of level representation (ignored)
            Returns (float): ratio of identical tiles to total tiles
        """
        return 1 - self.compare(a.data_chars, b.data_chars)
