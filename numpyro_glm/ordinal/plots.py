import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm


def plot_threshold_scatter(idata: az.InferenceData, thresholds: 'list[str]', ax: plt.Axes = None):
    """
    Scatter plot of thresholds sample from MCMC chains.

    Parameters
    ----------
    idata: az.InferenceData
        Inference Data returned from numpyro.
    thresholds: list[str]
        Name of the thresholds to be plotted.
        Assume that the first and the last are deterministic variables.
    ax: plt.Axes
        Matplotlib's axes for plotting.
    """
    posterior = idata['posterior']

    values = np.asarray([posterior[f'{t}'].values.flatten()
                         for t in thresholds])
    means = np.mean(values, axis=0)

    if ax is None:
        _, ax = plt.subplots()

    for vals in values:
        sns.scatterplot(x=vals, y=means, ax=ax)

    return ax


def plot_ordinal_data_with_posterior(
        idata: az.InferenceData, latent_mean: str, latent_sd: str, thresholds: 'list[str]',
        data: pd.DataFrame, ordinal_predicted='Y',
        ax: plt.Axes = None):
    if ax is None:
        _, ax = plt.subplots()

    # Plot data.
    sns.histplot(
        data, x=ordinal_predicted,
        bins=data[ordinal_predicted].cat.categories,
        stat='proportion',
        ax=ax)

    # Plot posterior distribution.
    posterior = idata['posterior']
    mean = posterior[latent_mean].values.flatten()
    sd = posterior[latent_sd].values.flatten()
    thres = np.asarray([posterior[f'{t}'].values.flatten()
                        for t in thresholds])
    cdf = norm.cdf((thres - mean[None, :]) / sd[None, :])
    probs = np.zeros((len(thresholds) + 1, mean.size), dtype=np.float32)
    probs[0] = cdf[0]
    probs[-1] = 1. - cdf[-1]
    probs[1:-1] = cdf[1:] - cdf[:-1]

    probs_mode = np.median(probs, axis=1)
    probs_95_hdi = az.hdi(probs.T - probs_mode[None, :], hdi_prob=.95)

    ax.errorbar(
        x=data[ordinal_predicted].cat.categories,
        y=probs_mode,
        yerr=np.abs(probs_95_hdi.T),
        fmt='o',
        elinewidth=3.)

    return ax
