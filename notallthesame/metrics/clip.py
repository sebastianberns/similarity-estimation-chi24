from pathlib import Path
from typing import Union

import torch
from torch import Tensor
from torch.nn.functional import cosine_similarity  # type: ignore[import]
from torchvision.transforms import ToTensor  # type: ignore[import]

from cleanfeatures import CleanFeatures  # type: ignore[import]

from ..level import Level
from ..metric import Metric


class CLIP(Metric):
    def __init__(self, model_path: Union[str, Path], clip_model: str = 'ViT-L/14@336px') -> None:
        super().__init__("CLIP")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cf = CleanFeatures(model_path, 'CLIP', self.device, log='warning', clip_model=clip_model)

    def level_to_tensor(self, level: Level, rep: str) -> Tensor:
        """ Convert level image to tensor """
        transform = ToTensor()
        img = level.load_img(rep)
        return transform(img)

    def similarity(self, a: Level, b: Level, rep: str) -> float:
        """ Calculate the similarity of two levels using CLIP
            Cosine similarity of the CLIP features

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
