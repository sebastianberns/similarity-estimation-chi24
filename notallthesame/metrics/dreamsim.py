from pathlib import Path
from typing import Union

import torch
from torch import Tensor
from torch.nn.functional import cosine_similarity  # type: ignore[import]
from torchvision.transforms import ToTensor  # type: ignore[import]

from cleanfeatures import CleanFeatures  # type: ignore[import]

from ..level import Level
from ..metric import Metric


class DreamSim(Metric):
    def __init__(self, model_path: Union[str, Path], dreamsim_type: str = 'ensemble') -> None:
        super().__init__("DreamSim")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cf = CleanFeatures(model_path, 'DreamSim', self.device, log='warning', dreamsim_type=dreamsim_type)

    def level_to_tensor(self, level: Level, rep: str) -> Tensor:
        """ Convert level image to tensor """
        transform = ToTensor()
        img = level.load_img(rep)
        return transform(img)

    def similarity(self, a: Level, b: Level, rep: str) -> float:
        """ Calculate the similarity of two levels using DreamSim
            Cosine similarity of the DreamSim features

            Arguments:
                a, b (Level):  Levels to compare
            Returns (float): similarity of two levels
        """
        ta = self.level_to_tensor(a, rep)
        tb = self.level_to_tensor(b, rep)
        input = torch.stack([ta, tb]).to(self.device)
        out = self.cf.compute_features(input)
        sim = cosine_similarity(out[0], out[1], dim=0)
        return sim.item()
