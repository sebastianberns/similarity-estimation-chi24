import numpy as np
from numpy.typing import NDArray

from ..level import Level
from ..metric import Metric


# Indices for Legend of Zelda (loz) 11x16 matrix
loz_idx_ut = (  # Upper triangle
    np.array([0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1, 2, 2, 2, 2, 2, 2, 2,  2, 3, 3, 3, 3, 3, 3,  3,  3, 4, 4, 4, 4, 4,  4, 5, 5, 5, 5,  5,  5, 6, 6, 6,  6, 7, 7,  7,  7, 8,  8,  9,  9]),
    np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 3, 4, 5, 6, 7, 8, 9, 10, 4, 5, 6, 7, 8, 9, 10, 11, 5, 6, 7, 8, 9, 10, 6, 7, 8, 9, 10, 11, 7, 8, 9, 10, 8, 9, 10, 11, 9, 10, 10, 11])
)
loz_idx_lt = (  # Lower triangle
    np.array([1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 3, 3, 5, 5, 7, 7, 9, 9, 4, 4, 6, 6, 8, 8, 10, 10, 5, 5, 7, 7, 9, 9, 6, 6, 8, 8, 10, 10, 7, 7, 9, 9, 8, 8, 10, 10, 9, 9, 10, 10]),
    np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 2, 1, 2, 1, 2, 1, 2,  1,  2, 2, 3, 2, 3, 2, 3, 2, 3, 3, 4, 3, 4, 3, 4,  3,  4, 4, 5, 4, 5, 4, 5, 5, 6, 5, 6,  5,  6, 6, 7, 6, 7, 7, 8,  7,  8, 8, 9,  9, 10])
)


class HorizontalSymmetry(Metric):
    """ Mirror across horizontal axis
            0 | 0
              |
            1 | 1
    """
    def __init__(self) -> None:
        super().__init__("Symmetry (Horizontal)")

    def ratio(self, data: NDArray) -> float:
        width = data.shape[1]  # Width of data
        half = width // 2  # Half of width

        left = data[:, :half]  # Left side values
        right = data[:, (width-half):]  # Right side values
        right = np.flip(right, axis=1)  # Flip right side horizontally

        match = left == right  # Compare left and right side values
        ratio = np.sum(match) / np.prod(left.shape)  # Ratio of matches to total number of values
        return ratio
    
    def similarity(self, a: Level, b: Level, rep: str) -> float:
        ra = self.ratio(a.data_chars)
        rb = self.ratio(b.data_chars)
        return np.abs(ra - rb)


class VerticalSymmetry(Metric):
    """ Vertical:  Mirror across vertical axis
            0  1
            ----
            0  1
    """
    def __init__(self) -> None:
        super().__init__("Symmetry (Vertical)")
        self.hs = HorizontalSymmetry()
    
    def ratio(self, data: NDArray) -> float:
        return self.hs.ratio(np.rot90(data))

    def similarity(self, a: Level, b: Level, rep: str) -> float:
        ra = self.ratio(a.data_chars)
        rb = self.ratio(b.data_chars)
        return np.abs(ra - rb)


class DiagonalFwdSymmetry(Metric):
    """ Diagonal / diagonal-forward:  Mirror across top-left to bottom-right diagonal axis
            0   1
              \
            1   0
    """
    def __init__(self) -> None:
        super().__init__("Symmetry (Diag Fwd)")

    def ratio(self, data: NDArray) -> float:
        h, w = data.shape  # Dimensions of data

        if h == w:  # Square matrix
            return self.ratio_square(data)

        assert h < w, "Non-square matrix need to b horizontally rectangular (n > m)"

        d = w - h  # Difference between width and height
        o = d // 2  # Offset
        r = d % 2  # Remainder

        # Remove offset on both sides
        data = data[:, o:-o]

        # If remainder r is 0, then treat as square matrix
        if r == 0:
            return self.ratio_square(data)

        if h == 11 and w == 16:
            return self.ratio_loz(data)
        
        print("TODO: general non-square matrix with remainder r = 1")
        return -1

    def ratio_square(self, data: NDArray) -> float:
        """ Diagonal / diagonal-fwd on square matrices """
        m, n = data.shape  # Dimensions of data
        assert m == n, "Input matrix ({m} x {n}) is not square"

        idx = np.triu_indices_from(data, k=1)  # Upper triangle indices (Tuple of NDArrays)
        upper = data[idx]  # Upper triangle values
        lower = data.T[idx]  # Lower triangle values

        match = upper == lower  # Compare upper and lower triangle values
        ratio = np.sum(match) / upper.size  # Ratio of matches to total number of values
        return ratio
    
    def ratio_loz(self, data: NDArray) -> float:
        upper = data[loz_idx_ut]  # Upper triangle values
        lower = data[loz_idx_lt]  # Lower triangle values

        match = upper == lower  # Compare upper and lower triangle values
        ratio = np.sum(match) / upper.size  # Ratio of matches to total number of values
        return ratio

    def similarity(self, a: Level, b: Level, rep: str) -> float:
        ra = self.ratio(a.data_chars)
        rb = self.ratio(b.data_chars)
        return np.abs(ra - rb)


class DiagonalBwdSymmetry(Metric):
    """ Anti-diagonal / diagonal-backward:  Mirror across bottom-left to top-right diagonal axis
            0   1
              /  
            1   0
    """
    def __init__(self) -> None:
        super().__init__("Symmetry (Diag Bwd)")
        self.fwd = DiagonalFwdSymmetry()

    def ratio(self, data: NDArray) -> float:
        return self.fwd.ratio(np.fliplr(data))

    def similarity(self, a: Level, b: Level, rep: str) -> float:
        ra = self.ratio(a.data_chars)
        rb = self.ratio(b.data_chars)
        return np.abs(ra - rb)
