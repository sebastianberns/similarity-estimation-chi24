from .level import Level


class Metric:
    def __init__(self, name: str) -> None:
        self.name = name

    def similarity(self, a: Level, b: Level, rep: str) -> float:
        """ Calculate the similarity of two levels
            Inverse of NCD(a, b)

            Arguments:
                a, b (Level):  Levels to compare
                rep (str):  Type of level representation
            Returns (float): NCD(a, b)
        """
        return 0.
