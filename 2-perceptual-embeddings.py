#!/usr/bin/env python
"""
Run t-STE on the judgement data 
for each experimental condition,
given the pre-optimized configurations, 
and save the resulting embeddings.
"""

import numpy as np

from notallthesame import config, get_embed_config, load_judgements_data, TSTE


print(f"Perceptual embeddings\n{config.embed_algorithm} ({config.embed_num_dims} dimensions)", end="\n\n")
for condition in config.conditions:
    print(f"Condition: {condition}...")

    judgements = load_judgements_data(condition)  # Load judgement data

    embed_config = get_embed_config(condition)  # Get embedding configuration
    params = embed_config['parameters']  # Get algorithm parameters

    tste = TSTE(**params['setup'], log_iter=0)
    embed, error, iter, num_viol = tste.run(judgements, **params['run'])
    num_constr = num_viol / tste.num_judgements
    print(f"Final iteration: {iter}, error: {error:.4f}, number of constraints: {num_constr:.4f}")

    save_path = config.embed_dir / f"{condition}-embed-{config.embed_algorithm}-{config.embed_num_dims}D.npz"
    np.savez(save_path, embedding=embed, config=embed_config)
    print(f"Saved embedding to {save_path}", end='\n\n')
