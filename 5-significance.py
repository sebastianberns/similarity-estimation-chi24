#!/usr/bin/env python
"""
Statistical signifiance tests
"""

from pathlib import Path
from typing import List

from matplotlib import pyplot as plt  # type: ignore[import]
import numpy as np
from numpy.typing import NDArray
import pandas as pd  # type: ignore[import]
from scipy.stats import f_oneway, ttest_rel, ttest_ind, false_discovery_control  # type: ignore[import]
from sklearn.metrics import cohen_kappa_score  # type: ignore[import]

from notallthesame import config


def test_statistical_significance(col: str, ttest_alt: str = "two-sided"):
    """
        col (str): Name of the statistic to test (column name)
    """
    save_names = {
        "Cohen’s kappa": "kappa",
        "Cohen’s kappa maximum": "kappa_max",
        "Unachieved agreement": "diff",
        "Quantity disagreement": "quant",
        "Allocation disagreement": "alloc",
    }

    df = pd.read_csv(config.save_dir / "agreement.csv", index_col=0)
    metrics = list(df['Metric'].unique())  # Get all metric names

    test_conditions(df, col, metrics)
    test_metrics_within(df, col, metrics, ttest_alt)
    test_metrics_between(df, col, metrics, ttest_alt)


def test_conditions(df, col, metrics):
    """
        One-way ANOVA in each condition.
        Test for the null hypothesis that several sets of samples have the same population mean.
        Null hypothesis can be rejected if p-value is less than a given significance level.
    """
    conditions = config.conditions
    pvalues = np.full((len(conditions), 1), np.nan, dtype=np.float64)
    for i, condition in enumerate(conditions):
        samples = df.loc[df['Condition']==condition][col].to_numpy()
        samples = samples.reshape((len(metrics), -1))
        pvalue = anova(samples)
        pvalues[i] = pvalue
    df = pd.DataFrame(data=pvalues, index=conditions, columns=["p-value"])
    df.to_csv(config.save_dir / f"significance_conditions-{get_save_name(col)}-pvalues.csv")


def test_metrics_within(df, col: str, metrics: List[str], ttest_alt: str = "two-sided"):
    """
        Paired Student's t-test within each condition (comparing different metrics in the same condition).
        Test for the null hypothesis that two related samples have identical average (expected) values.
        Null hypothesis can be rejected if p-value is less than a given significance level.
    """
    metrics_combinations_idxs = [(i, j) for i in range(len(metrics)) for j in range(i+1, len(metrics))]

    for condition in config.conditions:
        pvalues = np.full((len(metrics), len(metrics)), np.nan)

        for idx_a, idx_b in metrics_combinations_idxs:  # Test all combinations of metrics
            metric_a, metric_b = metrics[idx_a], metrics[idx_b]
            scores_a = get_scores(df, condition, metric_a, col)
            scores_b = get_scores(df, condition, metric_b, col)
            pvalue = paired_ttest(scores_a, scores_b, ttest_alt)
            pvalues[idx_a, idx_b], pvalues[idx_b, idx_a] = pvalue, pvalue

        save_results(pvalues, metrics, config.save_dir / f"significance_within-{get_save_name(col)}-{condition}-pvalues.csv")


def test_metrics_between(df, col: str, metrics: List[str], ttest_alt: str = "two-sided"):
    """
        Independent Student's t-test between each condition (comparing the same metrics in different conditions).
        Test for the null hypothesis that two independent samples have identical average (expected) values.
        Null hypothesis can be rejected if p-value is less than a given significance level.
    """
    conditions = config.conditions
    cond_combinations_idxs = [(i, j) for i in range(len(conditions)) for j in range(i+1, len(conditions))]

    for metric in metrics:
        # Collect p-values for all combinations of conditions
        pvalues_array = np.full((len(cond_combinations_idxs),), np.nan)

        for i, idxs in enumerate(cond_combinations_idxs):
            idx_a, idx_b = idxs
            cond_a, cond_b = conditions[idx_a], conditions[idx_b]
            scores_a = get_scores(df, cond_a, metric, col)
            scores_b = get_scores(df, cond_b, metric, col)
            pvalue = independent_ttest(scores_a, scores_b, ttest_alt)
            pvalues_array[i] = pvalue

        # Correct p-values for multiple comparisons
        pvalues_corrected = pvalue_correction(pvalues_array)

        # Save p-values in a symmetric matrix for pairwise comparison
        pvalues_matrix = np.full((len(conditions), len(conditions)), np.nan)
        for i, idxs in enumerate(cond_combinations_idxs):
            idx_a, idx_b = idxs
            pvalue = pvalues_corrected[i]
            pvalues_matrix[idx_a, idx_b], pvalues_matrix[idx_b, idx_a] = pvalue, pvalue

        save_results(pvalues_matrix, conditions, config.save_dir / f"significance_between-{get_save_name(col)}-{get_save_name(metric)}-pvalues.csv")


def get_scores(df: pd.DataFrame, condition: str, metric: str, col: str) -> NDArray:
    return df.loc[(df['Condition']==condition) & (df['Metric']==metric)][col].to_numpy()


def anova(samples: NDArray) -> float:
    """ One-way ANOVA """
    results = f_oneway(*samples, axis=0)
    pvalue = results[1]
    return pvalue


def paired_ttest(a: NDArray, b: NDArray, alternative: str = "two-sided") -> float:
    """ Paired student's t-test on two related samples """
    assert len(a) == len(b), "Both samples must have the same size."
    results = ttest_rel(a, b, alternative=alternative)
    pvalue = results[1]
    return pvalue


def independent_ttest(a: NDArray, b: NDArray, alternative: str = "two-sided") -> float:
    """ Student's t-test on two independent samples """
    assert len(a) == len(b), "Both samples must have the same size."
    results = ttest_ind(a, b, alternative=alternative)
    pvalue = results[1]
    return pvalue


def pvalue_correction(pvalues: NDArray, method: str = "bh") -> NDArray:
    """ Adjust p-values to control the false discovery rate (FDR), the 
    expected proportion of rejected null hypotheses that are actually true. """
    pvalues = false_discovery_control(pvalues, axis=None, method=method)
    return pvalues


def get_save_name(name: str) -> str:
    save_names = {
        "Cohen’s kappa": "kappa",
        "Cohen’s kappa maximum": "kappa_max",
        "Unachieved agreement": "diff",
        "Quantity disagreement": "quant",
        "Allocation disagreement": "alloc",

        "CLIP": "clip",
        "DreamSim": "dreamsim",
        "Normalised Compression Distance": "ncd",
        "Hamming Distance": "hamming",
        "Tile Frequencies": "tilefreq",
        "Tile Patterns (2×2)": "tilepat2",
        "Tile Patterns (3×3)": "tilepat3",
        "Tile Patterns (4×4)": "tilepat4",
        "Symmetry (Horizontal)": "symhoriz",
        "Symmetry (Vertical)": "symvert",
        "Symmetry (Diag Fwd)": "symdiagfwd",
        "Symmetry (Diag Bwd)": "symdiagbwd",
    }
    return save_names[name]


def save_results(data: NDArray, names: List[str], path: Path):
    df = pd.DataFrame(data=data, index=names, columns=names)
    df.to_csv(path)


if __name__ == "__main__":
    test_statistical_significance("Cohen’s kappa")
