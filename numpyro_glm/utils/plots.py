import arviz as az
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def plot_text(text: str, ax: plt.Axes):
    ax.text(0, 0, text, ha='center', fontsize='xx-large')
    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.25, 0.25)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_pairwise_scatter(mcmc, variables: 'list[str]', figsize: 'tuple[int, int]' = (10, 10), figtitle: str = None):
    nb_variables = len(variables)

    fig, axes = plt.subplots(
        nrows=nb_variables, ncols=nb_variables, figsize=figsize)
    fig.suptitle(figtitle)

    idata = az.from_numpyro(mcmc)
    posterior = idata.posterior

    for i, ith_var in enumerate(variables):
        for j, jth_var in enumerate(variables):
            ax = axes[j, i]

            if i == j:
                plot_text(ith_var, ax)
            elif i < j:
                ith_var_data = posterior[ith_var].values.flatten()
                jth_var_data = posterior[jth_var].values.flatten()

                corr, _ = pearsonr(ith_var_data, jth_var_data)
                plot_text(f'{corr:.2f}', ax)
            else:
                ith_var_data = posterior[ith_var].values.flatten()
                jth_var_data = posterior[jth_var].values.flatten()

                ax.scatter(ith_var_data, jth_var_data)

    fig.tight_layout()
    return fig


def plot_diagnostic(mcmc, variables: 'list[str]', detailed: bool = False):
    idata = az.from_numpyro(mcmc)

    for var in variables:
        if not detailed:
            az.plot_trace(idata, var_names=[var])
        else:
            print('WIP')

    return idata
