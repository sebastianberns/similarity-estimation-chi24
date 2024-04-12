#!/usr/bin/env python
"""
Given the raw study data file, 
parse the responses by condition, 
collect judgements as indices of stimuli (as they appear in the stimuli file)
and save judgements data.
"""

from pathlib import Path
import re
from typing import List

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from notallthesame import config, load_survey_data, load_triplets_data, load_stimuli_data


save_dir = config.judgements_dir


def parse_judgements(survey_file: Path, conditions: List[str]):
    print("Loading survey data...")
    survey_df = load_survey_data(survey_file)
    survey_cols = survey_df.columns  # All column names

    for condition in conditions:
        print(f"Condition: {condition}...")

        judgements = parse_responses_per_condition(survey_df, condition)
        num_judgements = len(judgements)
        
        package = {
            "judgements": judgements,
            "num_judgements": num_judgements,
            "num_stimuli": (np.max(judgements) + 1),
            "condition": condition,
            "description": "Survey responses by question and participants. Each row is a triplet judgement as stimuli indices (as they appear in the stimuli file). Columns are: prompt stimulus, stimulus selected by participant, remaining not selected stimulus. Each column is a participant."
        }
        save_file = save_dir / f"{condition}-judgements.npz"
        np.savez(save_file, **package)
        
        print(f"Saved {num_judgements} judgements ('{save_file}')", end='\n\n')


def parse_responses_per_condition(survey_df: DataFrame, condition: str) -> NDArray:
    triplets = load_triplets_data(condition)
    stimuli = load_stimuli_data(condition)

    re_cond = re.compile(condition + "-q[0-9]{1,3}")  # Condition questions pattern
    cond_cols = [c for c in survey_df.columns if re_cond.match(c)]  # Column names of condition questions

    cond_judgements_list: List[int] = []  # Dynamically growing list of judgements
    for col in cond_cols:  # Iterate over condition questions
        # Append question judgements to list
        cond_judgements_list += parse_responses_per_question(survey_df, col, triplets, stimuli)
    # Convert to numpy array and reshape to judgements
    cond_judgements = np.array(cond_judgements_list, dtype=np.int8).reshape((len(cond_judgements_list)//3, 3))
    return cond_judgements


def parse_responses_per_question(survey_df: DataFrame, col: str, triplets: DataFrame, stimuli: DataFrame) -> List[int]:
    answers = survey_df[col]  # All answers to question
    answers = answers.dropna()  # Filter not seen
    answers = answers[answers != "-99"]  # Filter seen but unanswered

    # Get question triplets: uids and indices
    name = col.split(' ')[0]  # Question name
    triplet_uids = get_triplet_stimuli_per_question(name, triplets)  # Get triplet stimuli uids
    triplet_idx = get_stimuli_indices_per_question(triplet_uids, stimuli)  # Get triplet indices in stimuli file
    assert len(triplet_uids) == 3, f"Question ‘{name}’ has {len(triplet_uids)} stimuli, expected 3"
    triplet_uid_to_idx = {uid: idx for uid, idx in zip(triplet_uids, triplet_idx)}  # Map triplet uid to index
    prompt_idx = triplet_idx[0]  # Index of question prompt

    # Collect triplet data
    question_triplets: List[int] = []
    for i, (_, stimulus) in enumerate(answers.items()):
        response_idx = triplet_uid_to_idx[stimulus]  # Index of stimulus chosen by participant

        remaining_triplets = triplet_idx[1:].copy()  # Get response indices
        remaining_triplets.remove(response_idx)  # Remove chosen stimulus index
        remaining_idx = remaining_triplets[0]  # Index of stimulus not chosen by participant

        question_triplets.append(prompt_idx)  # Index of question prompt
        question_triplets.append(response_idx)  # Index of stimulus chosen by participant
        question_triplets.append(remaining_idx)  # Index of stimulus not chosen by participant

    return question_triplets


def get_triplet_stimuli_per_question(question: str, triplets: DataFrame) -> List[str]:
    row = triplets[triplets['uid'] == question]  # Find triplet row
    cols = ['prompt', 'response1', 'response2']
    stimuli = row[cols].values.tolist()[0]  # Get stimuli
    return stimuli


def get_stimuli_indices_per_question(triplet_uids: List[str], triplets: DataFrame) -> List[int]:
    indices: List[int] = []
    for uid in triplet_uids:
        row = triplets['uid'] == uid  # Find stimulus row
        index = triplets.index[row].item()  # Get index of stimulus
        indices.append(index)
    return indices  # Return indices of stimuli


if __name__ == "__main__":
    print("Parsing human judgements from raw survey data")
    parse_judgements(config.survey_file, config.conditions)
