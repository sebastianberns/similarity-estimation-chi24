from io import BytesIO
from typing import Tuple
import zlib

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from PIL.Image import Image as PILImage

from ..level import Level
from ..metric import Metric


class NormalizedCompressionDistance(Metric):
    """ Normalized Compression Distance on tile representations """

    def __init__(self) -> None:
        super().__init__("Normalised Compression Distance")

    def compress(self, data: NDArray) -> bytes:
        """ Compress array using zlib
            Arguments:
                data (NDArray): Array to compress
            Returns (bytes): Compressed array
        """
        # input = data.flatten().tobytes()
        input = "".join(data.flatten().astype(str)).encode("utf-8")
        compression = zlib.compress(input, level=9)
        return compression
    
    def compression_len(self, data: NDArray) -> int:
        compression = self.compress(data)
        length = len(compression)
        return length
    
    def ncd(self, a: NDArray, b: NDArray) -> float:
        """ Normalized Compression Distance (NCD)
            Calculate the dissimilarity of two items:
                NCD(a, b) = (C(a + b) - min(C(a), C(b))) / max(C(a), C(b))
            where C(x) is the length of the compressed version of x

            Arguments:
                a, b (NDArray):  Arrays to compare
            Returns (float): Normalized Compression Distance between a and b
        """
        # Concatenate arrays
        ab = np.concatenate((a.flatten(), b.flatten()))

        # Get compression lengths
        la = self.compression_len(a)
        lb = self.compression_len(b)
        lab = self.compression_len(ab)

        # Calculate normalized compression distance
        distance = (lab - min(la, lb)) / max(la, lb)
        return distance
    
    def similarity(self, a: Level, b: Level, rep: str) -> float:
        """ Calculate the similarity of two levels
            Inverse of NCD(a, b)

            Arguments:
                a, b (Level):  Levels to compare
                rep (str):  Type of level representation (ignored)
            Returns (float): NCD(a, b)
        """
        distance = self.ncd(a.data_chars, b.data_chars)
        return 1 - distance


class NCDImages(Metric):
    """ Normalized Compression Distance on image representations """

    def __init__(self) -> None:
        super().__init__("Normalised Compression Distance (images)")

    def concat(self, images: Tuple[PILImage, ...]) -> PILImage:
        widths, heights = zip(*(i.size for i in images))

        max_width = max(widths)
        total_height = sum(heights)

        new = Image.new('RGB', (max_width, total_height))

        offset = 0
        for im in images:
            new.paste(im, (0, offset))
            offset += im.size[1]

        return new

    def compress(self, image: PILImage) -> bytes:
        """ Compress array using zlib
            Arguments:
                image (PILImage): Image to compress
            Returns (bytes): Compressed array
        """
        reader = BytesIO()
        image.save(reader, format='PNG')
        data = reader.getvalue()
        compression = zlib.compress(data, level=9)
        return compression
    
    def compression_len(self, data: PILImage) -> int:
        compression = self.compress(data)
        length = len(compression)
        return length
    
    def ncd(self, a: PILImage, b: PILImage) -> float:
        """ Normalized Compression Distance (NCD)
            Calculate the dissimilarity of two items:
                NCD(a, b) = (C(a + b) - min(C(a), C(b))) / max(C(a), C(b))
            where C(x) is the length of the compressed version of x

            Arguments:
                a, b (PILImage):  Images to compare
            Returns (float): Normalized Compression Distance between a and b
        """
        # Concatenate arrays
        ab = self.concat((a, b))

        # Get compression lengths
        la = self.compression_len(a)
        lb = self.compression_len(b)
        lab = self.compression_len(ab)

        # Calculate normalized compression distance
        distance = (lab - min(la, lb)) / max(la, lb)
        return distance
    
    def similarity(self, a: Level, b: Level, rep: str) -> float:
        """ Calculate the similarity of two levels
            Inverse of NCD(a, b)

            Arguments:
                a, b (Level):  Levels to compare
                rep (str):  Type of level representation (ignored)
            Returns (float): NCD(a, b)
        """
        distance = self.ncd(a.load_img(rep), b.load_img(rep))
        return 1 - distance
