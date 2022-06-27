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
        idata: az.InferenceData,
        latent_mean: str, latent_sd: str,
        thresholds: 'list[str]',
        data: pd.DataFrame, ordinal_predicted='Y',
        latent_coords: dict = None,
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
    mean = posterior[latent_mean].sel(latent_coords).values.flatten()
    sd = posterior[latent_sd].sel(latent_coords).values.flatten()
    thres = np.asarray([posterior[f'{t}'].values.flatten()
                        for t in thresholds])
    cdf = norm.cdf((thres - mean[None, :]) / sd[None, :])
    probs = np.zeros((len(thresholds) + 1, mean.size), dtype=np.float32)
    probs[0] = cdf[0]
    probs[-1] = 1. - cdf[-1]
    probs[1:-1] = cdf[1:] - cdf[:-1]

    probs_median = np.median(probs, axis=1)
    probs_95_hdi = az.hdi(probs.T - probs_median[None, :], hdi_prob=.95)

    ax.errorbar(
        x=data[ordinal_predicted].cat.categories,
        y=probs_median,
        yerr=np.abs(probs_95_hdi.T),
        fmt='o',
        elinewidth=3.)

    return ax


def plot_ordinal_data_with_linear_trend_and_posterior(
        idata: az.InferenceData, *,
        latent_intercept: str,
        latent_coef: str,
        latent_sigma: str,
        thresholds: 'list[str]',
        data: pd.DataFrame,
        ordinal_predicted: str,
        metric_predictor: str,
        nb_lin_trends: int = 20,
        ax: plt.Axes = None):
    if ax is None:
        _, ax = plt.subplots()

    # First, plot the data.
    sns.stripplot(x=metric_predictor, y=ordinal_predicted, data=data, ax=ax)

    # Obtain the MCMC samples.
    posterior = idata['posterior']
    b0 = posterior[latent_intercept].values.flatten()
    b1 = posterior[latent_coef].values.flatten()
    sigma = posterior[latent_sigma].values.flatten()
    thres = np.asarray([posterior[f'{t}'].values.flatten()
                        for t in thresholds])

    # Then, we will superimpose the linear trends.
    xrange = np.linspace(*ax.get_xlim(), 1000)
    trend_indices = np.random.choice(
        posterior['draw'].size * posterior['chain'].size,
        nb_lin_trends,
        replace=False
    )
    for idx in trend_indices:
        yrange = b0[idx] + b1[idx] * xrange
        # Minus 1 here because the axes vertical coordinate starts at 0,
        # so category 1 maps to 0, category 2 maps to 1.
        # And in our model, the first threshold is fixed at 1.5
        ax.plot(xrange, yrange - 1, c='b', alpha=.1)

    return ax
