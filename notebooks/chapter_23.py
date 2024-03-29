# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     notebook_metadata_filter: title, author
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   title: '[Doing Bayesian Data Analysis] Chapter 23: Ordinal Predicted Variable'
# ---

# %cd ..
# %load_ext autoreload
# %autoreload 2

# +
import arviz as az
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_median
import numpyro_glm.ordinal.models as glm_ordinal
import numpyro_glm.ordinal.plots as ordinal_plots
import pandas as pd
import seaborn as sns

numpyro.set_host_device_count(4)
# -

# # Chapter 23: Ordinal Predicted Variable
#
# ## The Case of A Single Group

ord_1_df: pd.DataFrame = pd.read_csv('datasets/OrdinalProbitData-1grp-1.csv')
yord_1_cat = pd.CategoricalDtype([1, 2, 3, 4, 5, 6, 7], ordered=True)
ord_1_df['Y'] = ord_1_df['Y'].astype(yord_1_cat)

kernel = NUTS(glm_ordinal.yord_single_group,
              init_strategy=init_to_median,
              target_accept_prob=0.999)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    y=jnp.array(ord_1_df['Y'].cat.codes.values),
    K=yord_1_cat.categories.size,
)
mcmc.print_summary()

idata_yord_1 = az.from_numpyro(mcmc)
az.plot_trace(idata_yord_1)
plt.tight_layout()


# +
def plot_ordinal_one_group_results(idata, df, thresholds):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))

    # Plot latent score's mean posterior distribution.
    ax = axes[0, 0]
    az.plot_posterior(
        idata,
        var_names='mu',
        hdi_prob=0.95,
        ref_val=2,
        point_estimate='mode',
        ax=ax)
    ax.set_title('Mean')
    ax.set_xlabel('$\mu$')

    # Plot data with posterior distribution.
    ax = axes[0, 1]
    ordinal_plots.plot_ordinal_data_with_posterior(
        idata, 'mu', 'sigma', thresholds,
        df, ordinal_predicted='Y',
        ax=ax)
    ax.set_title('Data with Post. Pred. Distrib.')

    # Plot latent score's standard deviation.
    ax = axes[1, 0]
    az.plot_posterior(
        idata,
        var_names='sigma',
        hdi_prob=0.95,
        point_estimate='mode',
        ax=ax)
    ax.set_title('Std.')
    ax.set_xlabel('$\sigma$')

    # Plot effect size.
    ax = axes[1, 1]
    posterior = idata.posterior
    eff_size = (posterior['mu'] - 2) / posterior['sigma']
    az.plot_posterior(
        eff_size,
        hdi_prob=0.95,
        point_estimate='mode',
        ref_val=0,
        ax=ax)
    ax.set_title('Effect Size')
    ax.set_xlabel('$(\mu - 2) / \sigma$')

    # Plot thresholds.
    ax = axes[2, 0]
    ordinal_plots.plot_threshold_scatter(idata, thresholds, ax=ax)

    axes[2, 1].remove()
    fig.tight_layout()


thresholds = ['thres_1', 'thres_2', 'thres_3', 'thres_4', 'thres_5', 'thres_6']
plot_ordinal_one_group_results(idata_yord_1, ord_1_df, thresholds)
# -

# ---
#
# The same model but applied on different dataset.

ord_2_df: pd.DataFrame = pd.read_csv('datasets/OrdinalProbitData-1grp-2.csv')
yord_2_cat = pd.CategoricalDtype([1, 2, 3, 4, 5, 6, 7], ordered=True)
ord_2_df['Y'] = ord_2_df['Y'].astype(yord_2_cat)

kernel = NUTS(glm_ordinal.yord_single_group,
              init_strategy=init_to_median,
              target_accept_prob=0.99)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    y=jnp.array(ord_2_df['Y'].cat.codes.values),
    K=yord_2_cat.categories.size,
)
mcmc.print_summary()

idata_yord_2 = az.from_numpyro(mcmc)
az.plot_trace(idata_yord_2)
plt.tight_layout()

plot_ordinal_one_group_results(idata_yord_2, ord_2_df, thresholds)

# ## The Case of Two Groups

two_groups_df: pd.DataFrame = pd.read_csv('datasets/OrdinalProbitData1.csv')
two_groups_ordinal_cat = pd.CategoricalDtype([1, 2, 3, 4, 5], ordered=True)
two_groups_df['Y'] = two_groups_df['Y'].astype(two_groups_ordinal_cat)
two_groups_df['X'] = two_groups_df['X'].astype('category')
two_groups_df.info()

kernel = NUTS(glm_ordinal.yord_two_groups,
              init_strategy=init_to_median,
              target_accept_prob=0.99)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    y=jnp.array(two_groups_df['Y'].cat.codes.values),
    grp=jnp.array(two_groups_df['X'].cat.codes.values),
    K=two_groups_ordinal_cat.categories.size,
    nb_groups=two_groups_df['X'].cat.categories.size,
)
mcmc.print_summary()

idata_two_groups = az.from_numpyro(
    mcmc,
    coords=dict(group=two_groups_df['X'].cat.categories),
    dims=dict(sigma=['group'], mu=['group'])
)
az.plot_trace(idata_two_groups)
plt.tight_layout()

# +
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12, 16))

# Plot group A's latent mean.
ax = axes[0, 0]
az.plot_posterior(
    idata_two_groups,
    var_names='mu',
    coords=dict(group='A'),
    point_estimate='mode',
    hdi_prob=0.95,
    ax=ax)
ax.set_title('A Mean')
ax.set_xlabel('$\mu_A$')

# Plot group A's data with posterior distrib.
ax = axes[0, 1]
ordinal_plots.plot_ordinal_data_with_posterior(
    idata_two_groups,
    'mu', 'sigma',
    ['thres_1', 'thres_2', 'thres_3', 'thres_4'],
    latent_coords=dict(group='A'),
    data=two_groups_df[two_groups_df['X'] == 'A'],
    ordinal_predicted='Y',
    ax=ax,
)
ax.set_title('Data for A with Post. Pred.')

# Plot group B's latent mean.
ax = axes[1, 0]
az.plot_posterior(
    idata_two_groups,
    var_names='mu',
    coords=dict(group='B'),
    point_estimate='mode',
    hdi_prob=0.95,
    ax=ax)
ax.set_title('B Mean')
ax.set_xlabel('$\mu_B$')

# Plot group B's data with posterior distrib.
ax = axes[1, 1]
thresholds = ['thres_1', 'thres_2', 'thres_3', 'thres_4']
ordinal_plots.plot_ordinal_data_with_posterior(
    idata_two_groups,
    'mu', 'sigma',
    thresholds,
    latent_coords=dict(group='B'),
    data=two_groups_df[two_groups_df['X'] == 'B'],
    ordinal_predicted='Y',
    ax=ax,
)
ax.set_title('Data for B with Post. Pred.')

# Plot A's standard deviation.
ax = axes[2, 0]
az.plot_posterior(
    idata_two_groups,
    var_names='sigma',
    coords=dict(group='A'),
    point_estimate='mode',
    hdi_prob=0.95,
    ax=ax)
ax.set_title('A Std.')
ax.set_xlabel('$\sigma_A$')

# Plot difference of means.
posterior = idata_two_groups['posterior']
ax = axes[2, 1]
mean_diff = (posterior['mu'].sel(group='B')
             - posterior['mu'].sel(group='A'))
az.plot_posterior(
    mean_diff,
    point_estimate='mode',
    hdi_prob=0.95,
    ax=ax)
ax.set_title('Difference of Means')
ax.set_xlabel('$\mu_B - \mu_A$')

# Plot B's standard deviation.
ax = axes[3, 0]
az.plot_posterior(
    idata_two_groups,
    var_names='sigma',
    coords=dict(group='B'),
    point_estimate='mode',
    hdi_prob=0.95,
    ax=ax)
ax.set_title('B Std.')
ax.set_xlabel('$\sigma_B$')

# Plot difference of sigmas.
ax = axes[3, 1]
sigma_diff = (posterior['sigma'].sel(group='B')
              - posterior['sigma'].sel(group='A'))
az.plot_posterior(
    sigma_diff,
    point_estimate='mode',
    hdi_prob=0.95,
    ax=ax)
ax.set_title('Difference of Sigmas')
ax.set_xlabel('$\sigma_B - \sigma_A$')

# Plot threshold scatter.
ax = axes[4, 0]
ordinal_plots.plot_threshold_scatter(
    idata_two_groups,
    thresholds,
    ax=ax
)
ax.set_xlabel('Threshold')
ax.set_ylabel('Mean Threshold')

# Plot effect size.
ax = axes[4, 1]
sigma_squared = (posterior['sigma'].sel(group='A')**2
                 + posterior['sigma'].sel(group='B')**2)
eff_size = mean_diff / np.sqrt(sigma_squared / 2)
az.plot_posterior(
    eff_size,
    point_estimate='mode',
    hdi_prob=0.95,
    ax=ax)
ax.set_title('Effect Size')
ax.set_xlabel('$(\mu_B - \mu_A) / \sqrt{\sigma_A^2 + \sigma_B^2}$')

fig.tight_layout()
# -

# ## The Case of Metric Predictors
# ### Example: Happiness and Money

happiness_cat = pd.CategoricalDtype([1, 2, 3, 4, 5], ordered=True)
happiness_df: pd.DataFrame = pd.read_csv(
    'datasets/HappinessAssetsDebt.csv',
    dtype={'Happiness': happiness_cat})
happiness_df.info()

happiness_fig, happiness_ax = plt.subplots()
sns.stripplot(
    x='Assets', y='Happiness',
    data=happiness_df,
    order=happiness_cat.categories[::-1],
    ax=happiness_ax)
happiness_fig.tight_layout()
print(happiness_ax.get_ylim())

kernel = NUTS(glm_ordinal.yord_metric_predictors,
              init_strategy=init_to_median)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=10000)
mcmc.run(
    random.PRNGKey(0),
    y=jnp.array(happiness_df['Happiness'].cat.codes.values),
    x=jnp.array(happiness_df[['Assets']].values),
    K=happiness_cat.categories.size,
)
mcmc.print_summary()

idata_happiness = az.from_numpyro(
    mcmc,
    coords=dict(pred=['Assets']),
    dims=dict(b=['pred'])
)
az.plot_trace(idata_happiness, var_names='~mu')
plt.tight_layout()

fig, ax = plt.subplots(figsize=(8, 6))
ordinal_plots.plot_ordinal_data_with_linear_trend_and_posterior(
    idata_happiness,
    latent_intercept='b0',
    latent_coef='b',
    latent_sigma='sigma',
    thresholds=['thres_1', 'thres_2', 'thres_3', 'thres_4'],
    data=happiness_df,
    ordinal_predicted='Happiness',
    metric_predictor='Assets',
    ax=ax
)
ax.invert_yaxis()
fig.tight_layout()

# +
fig: plt.Figure = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(nrows=2, ncols=3)

# Plot intercept posterior distribution.
ax: plt.Axes = fig.add_subplot(gs[0, 0])
az.plot_posterior(
    idata_happiness, 'b0', hdi_prob=.95, point_estimate='mode', ax=ax)
ax.set_title('Intercept')
ax.set_xlabel('$\\beta_0$')

# Plot coefficient for assets.
ax: plt.Axes = fig.add_subplot(gs[0, 1])
az.plot_posterior(
    idata_happiness,
    'b', coords=dict(pred='Assets'),
    hdi_prob=.95, point_estimate='mode',
    ax=ax)
ax.set_title('Assets')
ax.set_xlabel('$\\beta_1$')

# Plot standard deviation.
ax: plt.Axes = fig.add_subplot(gs[0, 2])
az.plot_posterior(
    idata_happiness,
    'sigma',
    hdi_prob=.95, point_estimate='mode',
    ax=ax)
ax.set_title('Std. Dev.')
ax.set_xlabel('$\\sigma$')

# Plot thresholds scatter.
ax: plt.Axes = fig.add_subplot(gs[1, :])
ordinal_plots.plot_threshold_scatter(
    idata_happiness,
    ['thres_1', 'thres_2', 'thres_3', 'thres_4'],
    ax=ax,
)
ax.set_xlabel('Threshold')
ax.set_ylabel('Mean Threshold')

fig.tight_layout()
# -

# ### Example: Movies - They don't make 'em like they used to

rating_cat = pd.CategoricalDtype([1, 2, 3, 4, 5, 6, 7], ordered=True)
movies_df: pd.DataFrame = pd.read_csv('datasets/Movies.csv')
movies_df['Rating'] = (movies_df['Rating']
                       .map({1: 1,
                             1.5: 2,
                             2: 3,
                             2.5: 4,
                             3: 5,
                             3.5: 6,
                             4: 7})
                       .astype(rating_cat))
movies_df.info()

fig_movies, ax_movies = plt.subplots(figsize=(8, 8))
sns.scatterplot(
    x='Year', y='Length',
    hue='Rating',
    style='Rating',
    markers={r: f'$\\mathbf{{{r}}}$' for r in rating_cat.categories},
    s=200,
    data=movies_df,
    legend=False,
    ax=ax_movies)
fig_movies.tight_layout()

kernel = NUTS(glm_ordinal.yord_metric_predictors,
              init_strategy=init_to_median,
              target_accept_prob=.95)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=20000, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    y=jnp.array(movies_df['Rating'].cat.codes.values),
    x=jnp.array(movies_df[['Year', 'Length']].values),
    K=rating_cat.categories.size,
)
mcmc.print_summary()

idata_movies = az.from_numpyro(
    mcmc,
    coords=dict(pred=['Year', 'Length']),
    dims=dict(b=['pred'])
)
az.plot_trace(idata_movies, '~mu')
plt.tight_layout()


# +
def superimpose_posterior_threshold_lines(
        idata: az.InferenceData, *,
        n_steps: int,
        latent_intercept: str,
        latent_coef: str,
        first_coord: dict,
        second_coord: dict,
        thresholds: 'list[str]',
        ax: plt.Axes):
    assert n_steps <= 3

    # Obtain the MCMC samples.
    posterior = idata['posterior']
    b0 = posterior[latent_intercept].values.flatten()
    b1 = posterior[latent_coef].sel(first_coord).values.flatten()
    b2 = posterior[latent_coef].sel(second_coord).values.flatten()
    thres = np.asarray([posterior[f'{t}'].values.flatten()
                        for t in thresholds])

    # Steps to plot.
    step_indices = np.random.choice(
        posterior['draw'].size * posterior['chain'].size,
        n_steps, replace=False)

    # Assumption: the y-axis will correspond with `b1`
    # and the x-axis will correspond with `b2`.
    yy, xx = np.meshgrid(
        np.linspace(*ax.get_ylim(), 1000),
        np.linspace(*ax.get_xlim(), 1000),
        indexing='ij',
    )

    linestyles = ['dashed', 'dotted', 'dashdot']
    for idx, ls in zip(step_indices, linestyles):
        mu = b0[idx] + b1[idx] * yy + b2[idx] * xx
        cs = ax.contour(
            xx, yy, mu,
            levels=thres[:, idx],
            colors='blue', alpha=.5, linestyles=ls)
        ax.clabel(
            cs, levels=thres[:, idx],
            fmt={lv: lb for lv, lb in zip(cs.levels, thresholds)})


superimpose_posterior_threshold_lines(
    idata_movies,
    n_steps=2,
    latent_intercept='b0',
    latent_coef='b',
    first_coord=dict(pred='Length'),
    second_coord=dict(pred='Year'),
    thresholds=['thres_1', 'thres_2', 'thres_3',
                'thres_4', 'thres_5', 'thres_6'],
    ax=ax_movies)

# Show the result.
fig_movies

# +
fig: plt.Figure = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(nrows=2, ncols=4)

# Plot intercept posterior distribution.
ax: plt.Axes = fig.add_subplot(gs[0, 0])
az.plot_posterior(
    idata_movies, 'b0', hdi_prob=.95, point_estimate='mode', ax=ax)
ax.set_title('Intercept')
ax.set_xlabel('$\\beta_0$')

# Plot coefficient for year.
ax: plt.Axes = fig.add_subplot(gs[0, 1])
az.plot_posterior(
    idata_movies,
    'b', coords=dict(pred='Year'),
    hdi_prob=.95, point_estimate='mode',
    ax=ax)
ax.set_title('Year')
ax.set_xlabel('$\\beta_{Year}$')

# Plot coefficient for length.
ax: plt.Axes = fig.add_subplot(gs[0, 2])
az.plot_posterior(
    idata_movies,
    'b', coords=dict(pred='Length'),
    hdi_prob=.95, point_estimate='mode',
    ax=ax)
ax.set_title('Length')
ax.set_xlabel('$\\beta_{Length}$')

# Plot standard deviation.
ax: plt.Axes = fig.add_subplot(gs[0, 3])
az.plot_posterior(
    idata_movies,
    'sigma',
    hdi_prob=.95, point_estimate='mode',
    ax=ax)
ax.set_title('Std. Dev.')
ax.set_xlabel('$\\sigma$')

# Plot thresholds scatter.
ax: plt.Axes = fig.add_subplot(gs[1, :])
ordinal_plots.plot_threshold_scatter(
    idata_movies,
    ['thres_1', 'thres_2', 'thres_3', 'thres_4', 'thres_5', 'thres_6'],
    ax=ax,
)
ax.set_xlabel('Threshold')
ax.set_ylabel('Mean Threshold')

fig.tight_layout()
