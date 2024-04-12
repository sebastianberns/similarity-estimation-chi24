from collections import Counter
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from ..distance import js_distance
from ..level import Level
from ..metric import Metric


class TileFrequencies(Metric):
    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__("Tile Frequencies")
        self.eps = eps  # Small value to avoid division by zero


    def frequencies(self, level: Level, tiles: Optional[List[str]] = None) -> NDArray:
        """ Calculate the frequencies of different tile types in a level """
        data = level.data_chars  # Level data
        if tiles is None:
            tiles = level.game_tiles  # Tile types for corresponding game

        num_tiles = data.size  # Number of tiles in level
        count = Counter(list(data.flatten()))  # Count number of tiles for different tile types

        freq = np.zeros(len(tiles), dtype=int)  # Initialize frequencies
        for i, tile in enumerate(tiles):
            if tile in count.keys():  # If tile type is present
                freq[i] = count[tile]  # Set count of tile type
        freq = (freq / num_tiles) + self.eps # Frequencies: normalize counts by number of tiles and add small value to avoid division by zero
        return freq


    def all_tiles(self, a: Level, b: Level) -> List[str]:
        """ Get combined list of all tile types in the two levels """
        assert a.game == b.game, "Levels must be from the same game"
        tiles = np.concatenate((a.data_chars.flatten(), b.data_chars.flatten()))
        tiles = np.unique(tiles)
        return tiles


    def shared_tiles(self, a: Level, b: Level) -> List[str]:
        """ Get list of tile types shared by two levels """
        assert a.game == b.game, "Levels must be from the same game"
        tiles = [t for t in a.game_tiles if t in a.data_chars and t in b.data_chars]
        return tiles


    def similarity(self, a: Level, b: Level, rep: str) -> float:
        """ Calculate the similarity wrt tile type frequencies of two levels

            Arguments:
                a, b (Level):  Levels to compare
                rep (str):  Type of level representation (ignored)
            Returns (float): similarity of two levels in terms of tile type frequencies
        """
        tiles = self.all_tiles(a, b)
        freq_a = self.frequencies(a, tiles)
        freq_b = self.frequencies(b, tiles)

        # Jensen-Shannon divergence
        js = js_distance(freq_a, freq_b)

        return 1 - js
