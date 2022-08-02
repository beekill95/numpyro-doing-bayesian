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
#   title: '[Doing Bayesian Data Analysis] Exercise 23.2'
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

# # Exercise 23.2
# ## (A) Thresholded cummulative-normal model with random guessing mixture

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


kernel = NUTS(glm_ordinal.yord_metric_predictors_robust_guessing,
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
fig: plt.Figure = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(nrows=2, ncols=5)

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

# Plot alpha.
ax: plt.Axes = fig.add_subplot(gs[0, 4])
az.plot_posterior(
    idata_movies,
    'alpha',
    hdi_prob=.95, point_estimate='mode',
    ax=ax)
ax.set_title('Alpha')
ax.set_xlabel('$\\alpha$')

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
# -

# __Is there anything unusual about the posterior distribution on the thresholds, and why?__
# Some sampled values of the thresholds are inverted
# (smaller thresholds have values larger than larger thresholds,
# for instance, the last two thresholds in the figure above).
# This is because when those values are sampled,
# in this model,
# the probability of a category will still be larger than zero because of the guessing distribution.
# However, for the normal model (without the guessing parameter),
# those values are not valid and not shown in the resulting chains.

# ## (B) Thresholded cummulative-$t$-distribution model

# +
# WIP: in order for this model to work,
# `betainc` function of jax has to have gradient w.r.t to its parameters
# (https://github.com/pyro-ppl/numpyro/issues/1452).
# kernel = NUTS(glm_ordinal.yord_metric_predictors_robust_t_dist,
#               init_strategy=init_to_median,
#               target_accept_prob=.90)
# mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=1)
# mcmc.run(
#     random.PRNGKey(0),
#     y=jnp.array(movies_df['Rating'].cat.codes.values),
#     x=jnp.array(movies_df[['Year', 'Length']].values),
#     K=rating_cat.categories.size,
# )
# mcmc.print_summary()

# +
# idata_t_dist = az.from_numpyro(
#     mcmc,
#     coords=dict(pred=['Year', 'Length']),
#     dims=dict(b=['pred'])
# )
# az.plot_trace(idata_t_dist, '~mu')
# plt.tight_layout()
