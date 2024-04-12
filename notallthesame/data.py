from pathlib import Path
from typing import List

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas import DataFrame

from . import config
from .level import Level


def load_survey_data(survey_file: Path) -> DataFrame:
    survey = pd.read_csv(survey_file, header=0, dtype="str")
    return survey


def load_triplets_data(condition: str) -> DataFrame:
    triplets_file = config.get_triplets_file(condition)
    triplets = pd.read_csv(triplets_file, index_col=0)  # Load triplets data
    return triplets


def load_stimuli_data(condition: str) -> DataFrame:
    stimuli_file = config.get_stimuli_file(condition)
    stimuli = pd.read_csv(stimuli_file, index_col=0)  # Load stimuli data
    return stimuli


def load_judgements_data(condition: str) -> NDArray:
    file_path = config.judgements_dir / f"{condition}-judgements.npz"
    data = np.load(file_path)
    judgements = data['judgements']
    return judgements


def get_embedding(condition: str, algorithm: str = config.embed_algorithm, 
                  num_dims: int = config.embed_num_dims) -> NDArray:
    file_path = config.embed_dir / f"{condition}-embed-{algorithm}-{num_dims}D.npz"
    data = np.load(file_path)
    embedding = data['embedding']
    return embedding


def get_levels(game: str) -> List[Level]:
    game_dir = config.level_dir / 'enc' / game
    file_names: List[Path] = sorted([f for f in game_dir.iterdir() if not f.name.startswith(".")])

    levels: List[Level] = []
    for file_name in file_names:
        name = file_name.stem
        level = Level(name, game.lower(), config.level_dir)
        levels.append(level)
    return levels
