#!/usr/bin/env python
"""
Inter-rater agreement analyses
"""

from pathlib import Path
from typing import List, Tuple
import re

from matplotlib import pyplot as plt  # type: ignore[import]
import numpy as np
from numpy.typing import NDArray
import pandas as pd  # type: ignore[import]
from ptitprince import RainCloud  # type: ignore[import]
import seaborn as sns  # type: ignore[import]
from sklearn.metrics import cohen_kappa_score  # type: ignore[import]

from notallthesame import (
    config, 
    load_survey_data, load_triplets_data, load_stimuli_data, get_metric_config, 
    cohen_kappa_max, quantity_disagreement, allocation_disagreement, 
    Level, get_levels, 
    Metric, metrics
)


def agreement_analysis(config):
    prec = config.table_prec  # Number of decimal places

    survey_df = load_survey_data(config.survey_file)  # Load survey data

    data_cols = ['Condition', 'Metric', 
                 'Cohen’s kappa', 'Cohen’s kappa maximum', 'Unachieved agreement', 
                 'Quantity disagreement', 'Allocation disagreement']
    data = pd.DataFrame(columns=data_cols)

    table = "\\begin{tabular}{l|lllll}\n"

    i = 0  # Row index
    for game in config.games:
        for rep in config.reps:
            condition = f"{game}-{rep}"
            print(f"{condition:<31}\tCohen's kappa   \tKappa max       \tKappa diff      \tQuantity disag. \tAllocation disag.")
            table += "\\textbf{" + condition + "}    & Cohen's kappa    & Kappa max        & Kappa diff.      & Quantity disag.  & Allocation disag. \\\\ \\hline\n"
            for metric_fn in metrics.__all__:
                metric_cfg = get_metric_config(metric_fn)
                metric = getattr(metrics, metric_fn)(**metric_cfg)  # Instantiate metric class
                print(f"{metric.name:<31}", end='\t')

                triplets = load_triplets_data(condition)  # Load condition triplet data
                stimuli = load_stimuli_data(condition)  # Load condition triplet data

                cond_cols = get_cond_cols(condition, survey_df)  # Get column names of condition questions
                cond_survey = survey_df[cond_cols]  # Filter survey to condition questions
                human_codes = encode_cond_survey(cond_survey, triplets)

                metric_codes = metric_triplets(metric, triplets, stimuli, game, rep, config.data_dir)

                kappas, kappas_max, quant, alloc = compare(human_codes, metric_codes)
                kappas_diff = kappas_max - kappas  # Cohen's difference

                kappas_mean, kappas_std = np.mean(kappas), np.std(kappas)
                kappas_max_mean, kappas_max_std = np.mean(kappas_max), np.std(kappas_max)
                kappas_diff_mean, kappas_diff_std = np.mean(kappas_diff), np.std(kappas_diff)
                quant_mean, quant_std = np.mean(quant), np.std(quant)
                alloc_mean, alloc_std = np.mean(alloc), np.std(alloc)

                # Print report
                print(f"{kappas_mean:.{prec}f} (±{kappas_std:.{prec}f})", end='\t')
                print(f"{kappas_max_mean:.{prec}f} (±{kappas_max_std:.{prec}f})", end='\t')
                print(f"{kappas_diff_mean:.{prec}f} (±{kappas_diff_std:.{prec}f})", end='\t')
                print(f"{quant_mean:.{prec}f} (±{quant_std:.{prec}f})", end='\t')
                print(f"{alloc_mean:.{prec}f} (±{alloc_std:.{prec}f})")

                table += f"{metric.name:<19} & {kappas_mean:.{prec}f} (±{kappas_std:.{prec}f}) & {kappas_max_mean:.{prec}f} (±{kappas_max_std:.{prec}f}) & {kappas_diff_mean:.{prec}f} (±{kappas_diff_std:.{prec}f}) & {quant_mean:.{prec}f} (±{quant_std:.{prec}f}) & {alloc_mean:.{prec}f} (±{alloc_std:.{prec}f}) \\\\\n"

                # Aggregate data
                temp = pd.DataFrame(columns=data_cols, data=np.hstack([
                    np.full((len(kappas), 1), condition),
                    np.full((len(kappas), 1), metric.name),
                    kappas.reshape(-1, 1),
                    kappas_max.reshape(-1, 1),
                    kappas_diff.reshape(-1, 1),
                    quant.reshape(-1, 1),
                    alloc.reshape(-1, 1)
                ]))
                data = pd.concat([data, temp], ignore_index=True)
            print()
            if condition != config.conditions[-1]:
                table += " & & & & & \\\\\n"
    table += "\\end{tabular}\n"

    # Convert to numeric data types
    data['Cohen’s kappa'] = pd.to_numeric(data['Cohen’s kappa'])
    data['Cohen’s kappa maximum'] = pd.to_numeric(data['Cohen’s kappa maximum'])
    data['Unachieved agreement'] = pd.to_numeric(data['Unachieved agreement'])
    data['Quantity disagreement'] = pd.to_numeric(data['Quantity disagreement'])
    data['Allocation disagreement'] = pd.to_numeric(data['Allocation disagreement'])

    # Plot and save
    data.to_csv(config.save_dir / "agreement.csv", index=True)
    save_latex_table(table, config.save_dir / "agreement.tex")

    data = config.rename_metrics(data)
    plot_rainclouds(data, "Cohen’s kappa", config.save_dir/"agreement-kappa-rainclouds.pdf")
    plot_rainclouds(data, "Unachieved agreement", config.save_dir/"agreement-diff-rainclouds.pdf")
    plot_rainclouds(data, "Quantity disagreement", config.save_dir/"agreement-quant-rainclouds.pdf")
    plot_rainclouds(data, "Allocation disagreement", config.save_dir/"agreement-alloc-rainclouds.pdf")
    plot_boxplots(data, "Cohen’s kappa", config.save_dir/"agreement-kappa-boxplots.pdf")
    plot_boxplots(data, "Unachieved agreement", config.save_dir/"agreement-diff-boxplots.pdf")
    plot_boxplots(data, "Quantity disagreement", config.save_dir/"agreement-quant-boxplots.pdf")
    plot_boxplots(data, "Allocation disagreement", config.save_dir/"agreement-alloc-boxplots.pdf")

    print(f"Saved results to '{config.save_dir}/'")
    print("- agreement.csv (Results data)")
    print("- agreement.tex (Latex table of results)")
    print("- agreement-kappa-rainclouds.pdf (Raincloud plots: Cohen’s kappa)")
    print("- agreement-diff-rainclouds.pdf (Raincloud plots: Unachieved agreement)")
    print("- agreement-quant-rainclouds.pdf (Raincloud plots: Quantity disagreement)")
    print("- agreement-alloc-rainclouds.pdf (Raincloud plots: Allocation disagreement)")
    print("- agreement-kappa-boxplots.pdf (Box plots: Cohen’s kappa)")
    print("- agreement-diff-boxplots.pdf (Box plots: Unachieved agreement)")
    print("- agreement-quant-boxplots.pdf (Box plots: Quantity disagreement)")
    print("- agreement-alloc-boxplots.pdf (Box plots: Allocation disagreement)")


def get_cond_cols(condition: str, survey_df: pd.DataFrame) -> List[str]:
    q = re.compile(condition + "-q[0-9]{1,3}")  # Question name pattern
    cond_cols = [c for c in survey_df.columns if q.match(c)]  # Column names of all questions
    return cond_cols


def encode_cond_survey(cond_survey: pd.DataFrame, triplets: pd.DataFrame) -> NDArray:
    """
    For a given condition,
    go through all corresponding triplets in the survey
    and encode the responses as categorical codes,
    where the codes are the indices of the response options.
    """
    cond_cols = cond_survey.columns  # Column names of all questions

    data = np.ones(cond_survey.shape, dtype=float) * -1  # Initialize data array
    for i, col in enumerate(cond_cols):  # Iterate over columns (i.e. questions, triplets)
        name = col.split(' ')[0]  # Get triplet name
        # Get response options for triplet
        categories = triplets[triplets['uid'] == name][['response1', 'response2']].to_numpy().flatten()
        # Convert answers to categorical codes
        data[:, i] = pd.Categorical(cond_survey[col], categories=categories).codes.astype(float)

    data[data == -1] = np.nan  # Replace missing data with NaN
    return data


def metric_triplets(metric: Metric, triplets: pd.DataFrame, stimuli: pd.DataFrame, game: str, rep: str, data_dir: Path) -> NDArray:
    """
    Given a metric, for each triplet in the given condition,
    compute the similarity between the prompt stimulus 
    and the two response stimuli.
    """
    levels = {level.name: level for level in get_levels(game)}

    def find_level_by_uid(uid: str) -> Level:
        """ Handle different file names for level representations """
        filename = stimuli[stimuli['uid'] == uid]['filename'].item()
        stem = Path(filename).stem

        # rep: img
        if stem.endswith('-1'):
            stem = stem.replace('-1', '-0')
        
        # rep: pat
        wong = re.compile("_wong_o0_[0-9]{1,3}")
        stem = wong.sub('', stem)
        
        return levels[stem]

    similarities = np.zeros((2, triplets.shape[0]), dtype=float)
    for i, row in triplets.iterrows():
        l = find_level_by_uid(row['prompt'])
        l0 = find_level_by_uid(row['response1'])
        l1 = find_level_by_uid(row['response2'])

        similarities[0, i] = metric.similarity(l, l0, rep)
        similarities[1, i] = metric.similarity(l, l1, rep)

    choices = np.argmax(similarities, axis=0, keepdims=True).astype(float)  # Get indices of most similar response stimulus
    return choices  # Return indices as categorical codes


def compare(human_codes: NDArray, metric_codes: NDArray, categories: List[int] = [0, 1]) -> Tuple[NDArray, ...]:
    assert human_codes.shape[1] == metric_codes.shape[1]
    num_participants, num_triplets = human_codes.shape

    kappas = np.zeros((num_participants,), dtype=float)
    kappas_max = np.zeros((num_participants,), dtype=float)
    quants = np.zeros((num_participants,), dtype=float)  # Quantity disagreement
    allocs = np.zeros((num_participants,), dtype=float)  # Allocation disagreement

    for i in range(num_participants):
        mask = np.logical_not(np.isnan(human_codes[i, :]))
        if np.sum(mask) > 0:
            data = np.vstack((human_codes[i, mask], metric_codes[:, mask]))
            # print(data.shape, data.dtype)
            # print(np.unique(data))
            # print(data)
            kappa = cohen_kappa_score(data[0, :], data[1, :], labels=categories)
            kappa_max = cohen_kappa_max(data[0, :], data[1, :], categories=categories)
            quant = quantity_disagreement(data[0, :], data[1, :], categories=categories)
            alloc = allocation_disagreement(data[0, :], data[1, :], categories=categories)
        else:
            kappa = np.nan
            kappa_max = np.nan
            quant = np.nan
            alloc = np.nan
        kappas[i] = kappa
        kappas_max[i] = kappa_max
        quants[i] = quant
        allocs[i] = alloc

    # Filter out NaNs
    def nan_filter(array: NDArray) -> NDArray:
        return array[np.logical_not(np.isnan(array))]

    return (nan_filter(kappas), nan_filter(kappas_max), 
            nan_filter(quants), nan_filter(allocs))


def axhlines(data, **kwargs):
    ys = kwargs.get('y', [1.5, 3.5, 7.5])
    ax = plt.gca()
    for y in ys:
        ax.axhline(y, color='lightgrey', linewidth=1, linestyle='-', zorder=0)


def plot_rainclouds(df: pd.DataFrame, col: str, save_path: Path) -> None:
    sns.set_style("ticks",{'axes.grid' : True})
    # Width: 7.058823529 * 0.85 = 6 inches
    g = sns.FacetGrid(df, col="Condition", col_wrap=2, height=7.058823529, aspect=0.85)
    g = g.map_dataframe(RainCloud, x="Metric", y=col, data=df, orient="h", 
                        palette='Set3', move=.2, point_size=2, box_notch=True, 
                        box_flierprops={"marker": "x"})
    g = g.map_dataframe(axhlines, y=[1.4, 3.4, 7.4])
    plt.savefig(save_path)
    plt.close()


def plot_boxplots(df: pd.DataFrame, col: str, save_path: Path) -> None:
    sns.set_style("ticks",{"axes.grid" : True})
    # g = sns.FacetGrid(df, col="Condition", col_wrap=4, height=5, aspect=1.2)
    # g = sns.FacetGrid(df, col="Condition", col_wrap=4, height=4.411764706, aspect=0.68)
    g = sns.FacetGrid(df, col="Condition", col_wrap=4, height=4.615384615, aspect=0.65)
    g = g.map_dataframe(sns.boxplot, y="Metric", x=col, data=df, orient="h", 
                        width=.3, order=None, color="Set3", palette="Set3", 
                        showcaps=True, dodge=False, notch=True, zorder=10,
                        showfliers=False, whis=0)
    g = g.map_dataframe(axhlines, y=[1.5, 3.5, 7.5])
    plt.savefig(save_path)
    plt.close()


def save_latex_table(table: str, save_path: Path) -> None:
    with open(save_path, 'wb+') as f:
        f.write(table.encode('utf8'))


if __name__ == "__main__":
    agreement_analysis(config)
