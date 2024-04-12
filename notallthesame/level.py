from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from PIL.Image import Image as PILImage

from . import config


class Level:
    def __init__(self, name: str, game: str, level_dir: Union[str, Path]) -> None:
        assert game in config.games

        self.name = name  # Name of level
        self.game = game  # Name of video game
        self.reps = ['img', 'pat']  # Representations of level

        level_dir = Path(level_dir)  # Path to level directory
        self.enc_dir = level_dir / "enc" / game  # Path to encoded level data directory
        self.img_dir = level_dir / "img" / game  # Path to image level data directory

        self.game_tiles = config.tile_types[self.game]  # Tile types for this game

        self.parse_char_data(self.load_enc())  # Level encoding data


    def load_enc(self) -> str:
        """ Load level encoding data from file """
        enc_file_path = self.enc_dir / f"{self.name}.txt"  # Path to level encoding file
        assert enc_file_path.exists(), f"File not found: {enc_file_path}"

        with open(enc_file_path) as f:
            text = f.read()  # Read whole file
        text = text.strip()  # Remove trailing whitespace
        return text
    

    def load_img(self, rep: str) -> PILImage:
        """ Load level image data in given representation from file """
        assert rep in self.reps, f"Invalid representation: {rep}"

        img_file_path = self.img_dir / rep / f"{self.name}.png"  # Path to level encoding file
        assert img_file_path.exists(), f"File not found: {img_file_path}"

        img = Image.open(img_file_path).convert('RGB')  # Load image in RGB format
        return img


    def parse_char_data(self, text: str) -> None: 
        """ Convert level encoding to representations (character, integer, one-hot) """
        rows = text.split('\n')  # Split into rows

        num_chars = len(text)  # Number of characters
        num_rows = len(rows)  # Number of rows
        num_cols = len(rows[0])  # Number of columns
        num_newlines = num_rows - 1  # Assumed number of line breaks
        num_tiles = num_chars - num_newlines  # Number of tiles
        assert num_tiles == (num_rows * num_cols), \
            f"Tiles in level {self.name} not filled: {(num_rows * num_cols) - num_tiles}"
        
        self.shape = (num_rows, num_cols)
        tiles_chars = np.empty(self.shape, dtype=object)
        tiles_ids = np.empty(self.shape, dtype=int)

        for i, row in enumerate(rows):
            for j, tile in enumerate(list(row)):
                assert tile in self.game_tiles, \
                    f"Tile type not recognized: {tile}"
                tiles_chars[i, j] = tile
                tiles_ids[i, j] = config.tile_type_indices[self.game][tile]

        self.data_chars = tiles_chars
        self.data_ids = tiles_ids
        self.data_onehot = self.one_hot(self.data_ids)

        self.tiles = [t for t in self.game_tiles if t in self.data_chars]


    def one_hot(self, a: NDArray) -> NDArray:
        """ One-hot level representation
        Adapted from https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
        """
        num_tile_types = len(self.game_tiles)
        num_tiles = a.size

        # Initialize output array, height and width dimensions flattened
        out = np.zeros((num_tile_types, num_tiles), dtype=np.uint8)

        # Fill in output array
        out[a.ravel(), np.arange(num_tiles)] = 1

        # Reshape to final dimensions [num_tile_types, height, width]
        out.shape = (num_tile_types,) + a.shape
        return out
