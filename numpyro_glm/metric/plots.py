import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import numpyro
from scipy.stats import norm


def plot_st(mcmc, y, figsize: 'tuple[int, int]' = (10, 10), figtitle: str = None):
    idata = az.from_numpyro(mcmc)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    fig.suptitle(figtitle)

    # Plot posterior mean.
    ax = axes[0, 0]
    ax.set_title('Mean')
    az.plot_posterior(idata, var_names=['mean'], point_estimate='mode', kind='hist', hdi_prob=0.95, ax=ax)
    ax.set_xlabel('$\mu$')

    # Plot data with posterior.
    ax = axes[0, 1]
    ax.set_title('Data w. posterior pred.')
    ax.hist(y, density=True)

    # Plot some posterior distributions.
    n_curves = 20
    samples_idx = np.random.choice(len(idata.posterior.chain) * len(idata.posterior.draw), n_curves, replace=False)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 1000)
    for idx in samples_idx:
        sample_mean = idata.posterior['mean'].values.flatten()[idx]
        sample_std = idata.posterior['std'].values.flatten()[idx]
        ax.plot(x, norm.pdf(x, sample_mean, sample_std), c='#87ceeb')

    ax.set_xlabel('y')

    # Plot standard deviation.
    ax = axes[1, 0]
    ax.set_title('Standard Deviation')
    az.plot_posterior(idata, var_names=['std'], point_estimate='mode', kind='hist', hdi_prob=0.95, ax=ax)
    ax.set_xlabel('$\sigma$')

    # Plot effect size.
    # TODO

    fig.tight_layout()
    return fig