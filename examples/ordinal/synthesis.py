# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: doing-bayes
#     language: python
#     name: doing-bayes
# ---

# %cd ../..
# %load_ext autoreload
# %autoreload 2

import numpy as np
import numpyro_glm.ordinal as glm_ordinal
import matplotlib.pyplot as plt


# # Ordinal Model with Synthesis Data
# ## Synthesis Data

# +
def create_ranking_votes(mean: float, std: float, thresholds: 'list[float]', N: int) -> 'list[int]':
    points = np.random.normal(mean, std, N)
    thresholds = [-np.inf, ] + thresholds + [np.inf, ]
    votes = [np.sum((prev <= points) & (points <= cur))
             for cur, prev in zip(thresholds[1:], thresholds[:-1])]
    return votes


def plot_ranking_votes(votes: 'list[int]', labels: 'list[str]') -> None:
    fig, ax = plt.subplots()
    ax.bar(range(len(labels)), votes, width=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    fig.tight_layout()


LATENT_MEAN = 1.0
LATENT_STD = 1.0

N = 4970
ORDINAL_VALUES = ['0 star', '1 star',
                  '2 stars', '3 stars', '4 stars', '5 stars']
THRESHOLDS = [0.5, 1.5, 2.5, 3.5, 4.5]

votes = create_ranking_votes(LATENT_MEAN, LATENT_STD, THRESHOLDS, N)
plot_ranking_votes(votes, ORDINAL_VALUES)
