#!/usr/bin/env python
"""
Comparison of similarity matrices
"""

from pathlib import Path
from typing import List

from matplotlib import pyplot as plt  # type: ignore[import]
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import seaborn as sns  # type: ignore[import]

from notallthesame import (
    config, 
    get_metric_config, 
    get_levels, 
    get_embedding, pairwise_embedding_similarity, 
    pairwise_level_similarity, 
    Metric, metrics
)


def similarity_analysis(config):
    print(f"Analysis: Comparison of Similarity Matrices", end='\n\n')

    scores = np.zeros((len(metrics.__all__), len(config.conditions)))
    metric_labels: List[str] = []  # Collect names of metric for plotting

    j = 0
    for game in config.games:
        for rep in config.reps:
            print(f"Condition: {game}-{rep}...")
            for i, metric_fn in enumerate(metrics.__all__):
                metric_config = get_metric_config(metric_fn)
                metric = getattr(metrics, metric_fn)(**metric_config)  # Instantiate metric class
                mse = compare(game, rep, metric, config.data_dir)
                scores[i, j] = mse
                if j == 0:
                    metric_labels.append(metric.name)
            j += 1

    print(scores)
    print()

    np.save(config.save_dir / 'similarity-mse.npy', scores)
    # plot_results(scores, config.conditions, metric_labels, config.save_dir)

    df = npy2df(scores, config.conditions, metric_labels)
    df = config.rename_metrics(df)
    plot_results(df, config.save_dir)
    
    table = generate_latex_table(scores, config.conditions, metric_labels, config.save_dir)
    print(table, end='\n\n')

    print(f"Saved results to '{config.save_dir}/'")
    print("- similarity-mse.npy (Numpy array)")
    print("- similarity-mse.pdf (Plot)")
    print("- similarity-mse.tex (Latex table)")


def compare(game: str, rep: str, metric: Metric, data_dir: Path):
    condition = f"{game}-{rep}"
    embedding = get_embedding(condition)

    levels = get_levels(game)  # Load levels

    embed_sim = pairwise_embedding_similarity(embedding)  # Pairwise similarities of embedded levels as baseline
    level_sim = pairwise_level_similarity(*levels, rep=rep, metric=metric)  # Pairwise similarities of levels for comparison

    assert embed_sim.shape == level_sim.shape, f"Similarity matrices have different shapes: {embed_sim.shape}, {level_sim.shape}"

    mse = np.sum(np.square(embed_sim - level_sim)) / embed_sim.size  # Mean squared error between similarity matrices
    return mse


def npy2df(scores, conditions, metric_labels):
    n_rows, n_cols = scores.shape

    cols = ['Condition', 'Metric', 'MSE']
    col_condition = np.repeat(conditions, n_rows).reshape((-1, 1))
    col_metric = np.tile(metric_labels, n_cols).reshape((-1, 1))
    col_scores = scores.transpose().reshape((-1, 1))

    df = pd.DataFrame(columns=cols, data=np.hstack([
        col_condition,
        col_metric,
        col_scores
    ]))
    df['MSE'] = df['MSE'].astype(float)
    return df


def plot_results(df: pd.DataFrame, save_dir: Path) -> None:
    def plot_hlines(data, **kwargs):
        ax = plt.gca()
        n = len(data)
        y = np.arange(n)
        a = np.zeros((n))
        b = data['MSE'].to_numpy()
        ax.hlines(y, a, b, linestyle='solid', color='grey', zorder=8)
        ax.set_xlim(0, None)

    def plot_axhlines(data, **kwargs):
        ys = kwargs.get('y', [1.5, 3.5, 7.5])
        ax = plt.gca()
        for y in ys:
            ax.axhline(y, color='lightgrey', linewidth=1, linestyle='-', zorder=0)

    def plot_grid(data, **kwargs):
        ax = plt.gca()
        ax.yaxis.grid(False) # Hide the horizontal gridlines
        ax.xaxis.grid(True) # Show the vertical gridlines

    g = sns.FacetGrid(df, col="Condition", col_wrap=4, height=4.615384615, aspect=0.65)
    g = g.map_dataframe(sns.scatterplot, y="Metric", x="MSE", data=df, s=80, color='black', zorder=9)
    g = g.map_dataframe(sns.scatterplot, y="Metric", x="MSE", data=df, hue="Metric", palette="Set3", s=40, zorder=10)
    g = g.map_dataframe(plot_hlines)
    g = g.map_dataframe(plot_axhlines, y=[1.5, 3.5, 7.5])
    g = g.map_dataframe(plot_grid)

    plt.savefig(save_dir / 'similarity-mse.pdf', bbox_inches='tight')
    plt.close()


def generate_latex_table(scores: NDArray, conditions: List[str], 
                         metric_labels: List[str], save_dir: Path) -> str:
    prec = config.table_prec  # Number of decimal places

    metrics_len = np.max([len(m) for m in metric_labels])

    table = "\\begin{table}[]\n\\begin{tabular}{l|llll}\n"
    table += f"{'':<{metrics_len}} & {' & '.join(conditions)}  \\\\ \\hline\n"
    for i, metric_fn in enumerate(metrics.__all__):
        metric_scores = '& '.join([f"{s:<8.{prec}f}" for s in scores[i, :]])
        table += f"{metric_labels[i]:<{metrics_len}} & {metric_scores} \\\\\n"
    table += "\\hline\n\\end{tabular}\n\\end{table}\n"

    with open(save_dir / 'similarity-mse.tex', 'wb+') as f:
        f.write(table.encode('utf8'))

    return table


if __name__ == "__main__":
    similarity_analysis(config)
