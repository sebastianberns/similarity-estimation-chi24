from numpy.typing import NDArray


def normalize(data: NDArray) -> NDArray:
    # Normalize to [0, 1]
    min, max = data.min(), data.max()
    data = (data - min)  # Shift to minimum of 0
    data = data / (max - min)  # Scale to [0, 1]
    return data
