from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from ..distance import kl_divergence, js_distance
from ..level import Level
from ..metric import Metric


class TilePatterns(Metric):
    def __init__(self, size: int = 2, eps: float = 1e-7) -> None:
        super().__init__(f"Tile Patterns ({size}×{size})")
        self.size = size  # Size of tile patterns
        self.eps = eps    # Small value to avoid division by zero

    def get_patterns_freq(self, data: NDArray) -> Tuple[NDArray, NDArray]:
        """ Retrieve all tile patterns and their frequencies from level data """
        s  = self.size
        v = data.shape[0] - s + 1  # Number of patterns on vertical axis
        h = data.shape[1] - s + 1  # Number of patterns on horizontal axis
        p = v * h                  # Total number of patterns

        patterns = np.zeros((p, s, s), dtype=data.dtype)

        idx = np.arange(s)     # Basic indices (e.g. [0, 1] for s = 2)
        m = np.repeat(idx, s)  # Basic vertical indices ([0, 0, 1, 1])
        n = np.tile(idx, s)    # Basic horizontal indices ([0, 1, 0, 1])

        k = 0  # Pattern index
        # Iterate over all patterns
        for i in range(v):  # Vertically
            for j in range(h):  # Horizontally
                y = m + i  # Vertical indices
                x = n + j  # Horizontal indices
                patterns[k, :, :] = data[y, x].reshape(s, s)
                k += 1

        patterns, counts = np.unique(patterns, axis=0, return_counts=True)
        assert np.sum(counts) == p, "Number of patterns does not match"
        freq = counts / np.sum(counts)  # Frequencies: normalize counts by number of patterns
        freq += self.eps  # Add small value to avoid division by zero
        return patterns, freq
    
    def pattern_to_str(self, pattern: NDArray) -> str:
        """ Convert a tile pattern to a string representation """
        return "".join(pattern.flatten().astype(str))
    
    def get_pat2freq(self, patterns: NDArray, freq: NDArray) -> Dict[str, float]:
        """ Create a mapping from tile patterns to the corresponding frequencies """
        assert len(patterns) == len(freq), "Number of patterns does not match number of frequencies"
        pat2freq = {}
        for i in range(len(patterns)):
            s = self.pattern_to_str(patterns[i])
            pat2freq[s] = freq[i]
        return pat2freq
    
    def joint_patterns_freq(self, a: NDArray, b: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        """ Get joint patterns and individual overall frequencies for two levels """
        # Get patterns and frequencies for both levels
        patterns_a, freq_a = self.get_patterns_freq(a)
        patterns_b, freq_b = self.get_patterns_freq(b)

        # Create mappings from patterns to frequencies
        pat2freq_a = self.get_pat2freq(patterns_a, freq_a)
        pat2freq_b = self.get_pat2freq(patterns_b, freq_b)

        # Combine patterns from both levels
        patterns = np.concatenate((patterns_a, patterns_b), axis=0)  
        patterns = np.unique(patterns, axis=0)  # Remove duplicates

        # Create new frequency vectors
        new_freq_a = np.full(len(patterns), self.eps)
        new_freq_b = np.full(len(patterns), self.eps)
        for i in range(len(patterns)):
            s = self.pattern_to_str(patterns[i])
            if s in pat2freq_a:
                new_freq_a[i] = pat2freq_a[s]
            if s in pat2freq_b:
                new_freq_b[i] = pat2freq_b[s]
        return patterns, new_freq_a, new_freq_b

    def similarity(self, a: Level, b: Level, rep: str) -> float:
        """ Calculate the similarity of two levels w.r.t. their tile pattern distributions

            Arguments:
                a, b (Level):  Levels to compare
                rep (str):  Type of level representation (ignored)
            Returns (float): similarity of two levels in terms of tile pattern distributions
        """
        patterns, freq_a, freq_b = self.joint_patterns_freq(a.data_ids, b.data_ids)
        distance = js_distance(freq_a, freq_b)
        sim = 1. - distance
        return sim


class TilePatterns1x1(TilePatterns):
    # Identical to TileFrequencies
    def __init__(self) -> None:
        super().__init__(size=1)


class TilePatterns2x2(TilePatterns):
    def __init__(self) -> None:
        super().__init__(size=2)


class TilePatterns3x3(TilePatterns):
    def __init__(self) -> None:
        super().__init__(size=3)


class TilePatterns4x4(TilePatterns):
    def __init__(self) -> None:
        super().__init__(size=4)
