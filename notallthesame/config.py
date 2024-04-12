import json
from pathlib import Path
from typing import Dict, Union

import pandas as pd

from .metric import Metric


""" Paths and directories """
# Code base directory (will be loaded as a submodule, so this is relative to the parent directory)
base_dir = Path('.').resolve()

code_dir = base_dir / 'notallthesame'  # Code directory
model_dir = code_dir / 'models'  # Model directory
data_dir = base_dir / 'data'  # Data directory

survey_dir = data_dir / 'survey'
judgements_dir = data_dir / 'judgements'
embed_dir = data_dir / 'embeddings'
embed_config_dir = embed_dir / 'configs'
triplets_dir = data_dir / 'triplets'
stimuli_dir = data_dir / 'stimuli'
level_dir = data_dir / 'levels'

save_dir = base_dir / 'results'  # Save directory


""" Files """
survey_file = survey_dir / 'qualtrics-data.csv'

def get_triplets_file(condition: str) -> Path:
    return triplets_dir / f'{condition}-triplets.csv'

def get_stimuli_file(condition: str) -> Path:
    return stimuli_dir / f'{condition}-stimuli.csv'


""" Experimental conditions """
games = ['ccs', 'loz']
reps = ['img', 'pat']
conditions = ['ccs-img', 'ccs-pat', 'loz-img', 'loz-pat']


""" Levels """
# Mapping from tile type to tile name per game
tile_names = {
    'loz': {  # 10 tiles
        "-": "Hole",
        "F": "Floor",
        "B": "Block",
        "M": "Monster",
        "P": "Element",  # e.g., water or lava
        "O": "Elemental floor",
        "I": "Elemental block",
        "D": "Door",
        "S": "Stairs",
        "W": "Walls"
    },

    'ccs': {  # 7 tiles
        # Levels were translated from sprites in this order of priority as well.
        # Since there could be multiple things at a given position, we took the most "important"
        "X": "Void",  # basically what makes the shape of the level, the blank spots where nothing can go
        "B": "Blocker",
        "S": "Special Candy",  # like power pieces, bombs, etc.
        "L": "Lock",  # placed overtop of things and need to be cleared
        "J": "Jelly",  # placed underneath things and need to collected/cleared
        "R": "Regular Candy",  # just a normal candy piece
        "E": "Empty"  # a blank space in the level where candies and the like can fall into, but isnt currently occupied
    }
}

# Mapping from index to tile type
tile_types = {
    'loz': list(tile_names['loz'].keys()),
    'ccs': list(tile_names['ccs'].keys())
}

# Mapping from tile type to index
tile_type_indices = {
    'loz': {k: i for i, k in enumerate(tile_types['loz'])},
    'ccs': {k: i for i, k in enumerate(tile_types['ccs'])},
}


""" Embeddings """
embed_algorithm = 'tste'
embed_num_dims = 4

def get_embed_config(condition: str, algorithm: str = embed_algorithm, 
                     num_dims: int = embed_num_dims) -> Dict[str, Dict[str, Union[str, int, float, bool]]]:
    embed_config_file = embed_config_dir / f"{condition}-{algorithm}-{num_dims}D.json"
    with open(embed_config_file) as f:
        embed_config = json.load(f)
    return embed_config


""" Metrics """
def get_metric_config(metric: Metric) -> Dict[str, Union[str, int, Path]]:
    if metric == 'CLIP':
        return {
            'model_path': model_dir / 'clip',
            'clip_model': 'ViT-L/14@336px'
        }
    elif metric == 'DreamSim':
        return {
            'model_path': model_dir / 'dreamsim',
            'dreamsim_type': 'ensemble'
        }
    return {}


""" Latex tables """
table_prec = 3  # Number of decimal places


""" Plots """
metrics_formatted = {
    'CLIP': 'CLIP',
    'DreamSim': 'DreamSim',
    'Normalised Compression Distance': 'NCD',
    'Hamming Distance': 'Hamming',
    'Tile Frequencies': 'Tile Freq',
    'Tile Patterns (2×2)': 'Tile Pat (2×2)',
    'Tile Patterns (3×3)': 'Tile Pat (3×3)',
    'Tile Patterns (4×4)': 'Tile Pat (4×4)',
    'Symmetry (Horizontal)': 'Sym (Horiz)',
    'Symmetry (Vertical)': 'Sym (Vert)',
    'Symmetry (Diag Fwd)': 'Sym (Diag Fwd)',
    'Symmetry (Diag Bwd)': 'Sym (Diag Bwd)',
}

def rename_metrics(df: pd.DataFrame) -> pd.DataFrame:
    for old, new in metrics_formatted.items():
        df['Metric'] = df['Metric'].replace(old, new)
    return df
