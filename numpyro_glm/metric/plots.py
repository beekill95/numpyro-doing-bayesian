import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, t


def plot_st(mcmc, y,
            mean_comp_val: float = None, mean_ROPE: 'tuple[float, float]' = None,
            std_comp_val: float = None, std_ROPE: 'tuple[float, float]' = None,
            effsize_comp_val: float = None, effsize_ROPE: 'tuple[float, float]' = None,
            point_estimate: str = 'mode',
            HDI: float = 0.95,
            n_posterior_curves: int = 20,
            figsize: 'tuple[int, int]' = (10, 6), figtitle: str = None):
    idata = az.from_numpyro(mcmc)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    fig.suptitle(figtitle)

    # Plot posterior mean.
    ax = axes[0, 0]
    az.plot_posterior(
        idata,
        ref_val=mean_comp_val,
        rope=mean_ROPE,
        var_names=['mean'],
        point_estimate=point_estimate,
        kind='hist',
        hdi_prob=HDI,
        ax=ax)
    ax.set_title('Mean')
    ax.set_xlabel('$\mu$')

    # Plot data with posterior.
    ax = axes[0, 1]
    ax.set_title('Data w. posterior pred.')
    ax.hist(y, density=True)

    # Plot some posterior distributions.
    samples_idx = np.random.choice(
        len(idata.posterior.chain) * len(idata.posterior.draw),
        n_posterior_curves,
        replace=False)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 1000)
    for idx in samples_idx:
        sample_mean = idata.posterior['mean'].values.flatten()[idx]
        sample_std = idata.posterior['std'].values.flatten()[idx]
        ax.plot(x, norm.pdf(x, sample_mean, sample_std), c='#87ceeb')

    ax.set_xlabel('y')

    # Plot standard deviation.
    ax = axes[1, 0]
    az.plot_posterior(
        idata,
        ref_val=std_comp_val,
        rope=std_ROPE,
        var_names=['std'],
        point_estimate=point_estimate,
        kind='hist',
        hdi_prob=HDI,
        ax=ax)
    ax.set_title('Standard Deviation')
    ax.set_xlabel('$\sigma$')

    # Plot effect size.
    ax = axes[1, 1]
    if mean_comp_val is None:
        ax.remove()
    else:
        az.plot_posterior(
            (idata.posterior['mean'] - mean_comp_val) / idata.posterior['std'],
            ref_val=effsize_comp_val,
            rope=effsize_ROPE,
            point_estimate=point_estimate,
            kind='hist',
            hdi_prob=HDI,
            ax=ax)
        ax.set_title('Effect Size')
        ax.set_xlabel(f'$(\mu - {mean_comp_val}) / \sigma$')

    fig.tight_layout()
    return fig


def plot_st_2(mcmc, y,
              mean_comp_val: float = None, mean_ROPE: 'tuple[float, float]' = None,
              sigma_comp_val: float = None, sigma_ROPE: 'tuple[float, float]' = None,
              effsize_comp_val: float = None, effsize_ROPE: 'tuple[float, float]' = None,
              point_estimate: str = 'mode',
              HDI: float = 0.95,
              n_posterior_curves: int = 20,
              figsize: 'tuple[int, int]' = (15, 6), figtitle: str = None):
    idata = az.from_numpyro(mcmc)
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=figsize)
    fig.suptitle(figtitle)

    # Plot posterior mean.
    ax = axes[0, 0]
    az.plot_posterior(
        idata,
        ref_val=mean_comp_val,
        rope=mean_ROPE,
        var_names=['mean'],
        point_estimate=point_estimate,
        kind='hist',
        hdi_prob=HDI,
        ax=ax)
    ax.set_title('Mean')
    ax.set_xlabel('$\mu$')

    # Plot data with posterior.
    ax = axes[0, 1]
    ax.set_title('Data w. posterior pred.')
    ax.hist(y, density=True)

    # Plot some posterior distributions.
    samples_idx = np.random.choice(
        len(idata.posterior.chain) * len(idata.posterior.draw),
        n_posterior_curves,
        replace=False)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 1000)
    for idx in samples_idx:
        sample_mean = idata.posterior['mean'].values.flatten()[idx]
        sample_sigma = idata.posterior['sigma'].values.flatten()[idx]
        sample_nu = idata.posterior['nu'].values.flatten()[idx]
        ax.plot(x, t.pdf(x, loc=sample_mean, scale=sample_sigma, df=sample_nu), c='#87ceeb')

    ax.set_xlabel('y')

    # Plot standard deviation.
    ax = axes[1, 0]
    az.plot_posterior(
        idata,
        ref_val=sigma_comp_val,
        rope=sigma_ROPE,
        var_names=['sigma'],
        point_estimate=point_estimate,
        kind='hist',
        hdi_prob=HDI,
        ax=ax)
    ax.set_title('Sigma')
    ax.set_xlabel('$\sigma$')

    # Plot effect size.
    ax = axes[1, 1]
    if mean_comp_val is None:
        ax.remove()
    else:
        az.plot_posterior(
            (idata.posterior['mean'] - mean_comp_val) / idata.posterior['sigma'],
            ref_val=effsize_comp_val,
            rope=effsize_ROPE,
            point_estimate=point_estimate,
            kind='hist',
            hdi_prob=HDI,
            ax=ax)
        ax.set_title('Effect Size')
        ax.set_xlabel(f'$(\mu - {mean_comp_val}) / \sigma$')

    # Plot normality parameter.
    ax = axes[2, 0]
    az.plot_posterior(
        np.log(idata.posterior['nu']),
        point_estimate=point_estimate,
        kind='hist',
        hdi_prob=HDI,
        ax=ax)
    ax.set_title('Normality-Log')
    ax.set_xlabel('$log(\\nu$)')

    # Remove the last axes.
    ax = axes[2, 1]
    ax.remove()

    fig.tight_layout()
    return fig