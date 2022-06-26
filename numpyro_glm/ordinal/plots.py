import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_threshold_scatter(idata: az.InferenceData, thresholds: 'list[str]', ax: plt.Axes = None):
    """
    Scatter plot of thresholds sample from MCMC chains.

    Parameters
    ----------
    idata: az.InferenceData
        Inference Data returned from numpyro.
    thresholds: list[str]
        Name of the thresholds to be plotted.
        Assume that the first and the last are deterministic variable.
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
