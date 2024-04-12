# Not All the Same: Understanding and Informing Similarity Estimation in Tile-Based Video Games (CHI 2024)

Official Python implementation, all data and results. To reproduce results from raw data to paper plots and tables follow instructions below.

## Installation

1. Setup a Python environment (tested with Python 3.8.5)
1. Install dependencies
   `pip install -r requirements.txt`
1. Run scripts
   - either individually
     e.g. `python 3-similarity.py` (see list of contents below)
   - or all together
     `./run.sh`

For fastest execution a CUDA compatible GPU is recommended. With enough memory, the scripts should also run on CPU.

## Contents

- Implementation
  `notallthesame/`
  - All measures, embeddings and metrics
  - t-STE
  - Cohen’s Kappa maximum value
  - Quantity and Allocation Disagreement
  
- Scripts
  1. Parsing of human judgements from raw survey data
     `1-parse-judgements.py`
  1. Embedding of stimuli in perceptual spaces
     `2-perceptual-embeddings.py`
  1. Comparison of similarity matrices
     `3-similarity.py`
  1. Inter-rater agreement analyses
     `4-agreement.py`
  1. Statistical significance tests
     `5-significance.py`
- Data
  - Raw survey data from Qualtrics (anonymised and filtered)
    `data/survey/qualtrics-data.csv`
  - Level data as images and tile encodings in all experimental conditions
    `data/levels/`
  - List of stimuli (mapping file names to ids)
    `data/stimuli/`
  - List of triplet (identifying combinations of triplets by id)
    `data/triplets/`
  - Configurations of t-STE embedding algorithm
    `data/embeddings/configs/`
  - Parsed judgements data (script 1)
    `data/judgements/`
  - Perceptual embeddings (script 2)
    `data/embeddings/`
- Results
  - Similarity analysis (script 3)
    - Data as Numpy array
      `results/similarity-mse.npy`
    - Plot of MSE
      `results/similarity-mse.pdf`
    - Latex table of MSE
      `results/similarity-mse.tex`
  - Agreement analyses (script 4)
    - Table of results
      `results/agreement.csv `
    - Plot: Cohen’s kappa
      `results/agreement-kappa.pdf`
    - Plot: Difference maximum kappa and kappa
      `results/agreement-diff.pdf`
    - Plot: Quantity disagreement
      `results/agreement-quant.pdf`
    - Plot: Allocation disagreement
      `results/agreement-alloc.pdf`
  - Statistical significance (script 5)
    - P-values of one-way ANOVA of Cohen’s kappa in each condition
      `significance_conditions-kappa-pvalues.csv`
    - P-values of paired Student's t-test within each condition (comparing different metrics in the same condition)
      `significance_within-kappa-ccs-img-pvalues.csv ...`
    - P-values of independent Student's t-test between each condition (comparing the same metrics in different conditions)
      `significance_between-kappa-clip-pvalues.csv ...`

## Citation

Berns, S., Volz, V., Tokarchuk, L., Snodgrass, S., & Guckelsberger, C. (2024). Not All the Same: Understanding and Informing Similarity Estimation in Tile-Based Video Games. In *Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems*.

```
@inproceedings{berns2024not,
  title={Not All the Same: Understanding and Informing Similarity Estimation in Tile-Based Video Games},
  author={Berns, Sebastian and Volz, Vanessa and Tokarchuk, Laurissa and Snodgrass, Sam and Guckelsberger, Christian},
  booktitle={Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems},
  year={2024}
}
```
